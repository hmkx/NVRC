"""
NVRC main script
"""
from main_utils import *


def main():
    total_start_time = time.time()

    # Set & store args
    args = parse_args()

    # Set accelerator, logger, output dir
    accelerator, logger, output_dir = get_accelerator_logger(args)
    logger.info(f'Output dir: {output_dir}')
    temp_dir = os.path.join(output_dir, 'tmp')
    os.makedirs(temp_dir)

    # Set seed
    logger.info(f'Set seed: {args.seed}')
    accelerate.utils.set_seed(args.seed)

    # Create input/output videos
    train_video_in, train_video_out = create_encoding_decoding_videos(temp_dir, args, logger, training=True)
    eval_video_in, eval_video_out = create_encoding_decoding_videos(temp_dir, args, logger, training=False)
    assert all(train_video_in.get_video_size()[d] == eval_video_in.get_video_size()[d] for d in range(3)), \
        f'Input videos have different sizes: {train_video_in.get_video_size()} vs {eval_video_in.get_video_size()}'

    # Create dataset
    train_dataset = create_overfit_dataset(args, logger, train_video_in, channel_scale=None, channel_shift=None)
    eval_dataset = create_overfit_dataset(args, logger, eval_video_in, channel_scale=None, channel_shift=None)

    # Create tasks
    train_task = create_overfit_task(args, logger, train_video_out, channel_scale=None, channel_shift=None,
                                     training=True, device=accelerator.device)
    eval_task = create_overfit_task(args, logger, eval_video_out, channel_scale=None, channel_shift=None,
                                    training=False, device=accelerator.device)

    # Set coding settings
    assert args.start_frame == -1 or \
           (args.start_frame >= 0 and args.start_frame < train_dataset.get_video_size()[0])
    start_frame = args.start_frame if args.start_frame != -1 else 0
    assert args.num_frames == -1 or \
           (args.num_frames > 0 and args.num_frames <= train_dataset.get_video_size()[0] - start_frame)
    num_frames = args.num_frames if args.num_frames != -1 else train_dataset.get_video_size()[0] - start_frame
    assert args.intra_period == -1 or args.intra_period > 0
    intra_period = min(args.intra_period, num_frames) if args.intra_period != -1 else num_frames

    logger.info(f'Encoding settings:')
    logger.info(f'     Start frame - {start_frame}    Num of frames - {num_frames}')
    logger.info(f'     Intra period - {intra_period}')

    intra_idx = 0
    intra_idx_max = (num_frames + intra_period - 1) // intra_period

    # Create bitstream
    bs = b''

    # Create buffers
    all_num_frames = []
    all_size = []
    all_metrics = []

    while intra_idx < intra_idx_max:
        intra_start_frame = start_frame + intra_idx * intra_period
        intra_num_frames = min(intra_period, start_frame + num_frames - intra_start_frame)
        intra_bs = b''

        # Print group info
        logger.info(f'\nStart encoding:')
        logger.info(f'     Group index - {intra_idx}    Group start frame - {intra_start_frame}   Group num of frames - {intra_num_frames}')

        # Set group of frames for coding
        for dataset in [train_dataset, eval_dataset]:
            dataset.set_frames(intra_start_frame, intra_num_frames)
            dataset.create_cache()

        for task in [train_task, eval_task]:
            task.set_frames(intra_start_frame, intra_num_frames)
            task.create_cache()

        # Create loader
        train_loader = create_overfit_loader(args, logger, accelerator, train_dataset,
                                             start_frame=0, num_frames=intra_num_frames,
                                             training=True)
        eval_loader = create_overfit_loader(args, logger, accelerator, eval_dataset,
                                            start_frame=0, num_frames=intra_num_frames,
                                            training=False)

        # Create codec
        model = create_codec(output_dir, f'{intra_idx:04d}', args, logger, accelerator,
                                eval_loader, eval_task)

        # Optimizer & scheduler
        optimizer, scheduler = create_optimizer_scheduler(output_dir, f'{intra_idx:04d}', args, logger,
                                                          accelerator, model, train_loader)

        # Place model etc. to accelerator
        model, train_loader, eval_loader, optimizer, scheduler = \
            accelerator.prepare(model, train_loader, eval_loader, optimizer, scheduler)

        # Profile codec
        num_params, num_params_codec, num_macs, num_macs_rate = \
            profile_codec(output_dir, f'{intra_idx:04d}', args, logger, accelerator, model, eval_loader, eval_task)

        # Restoring training state
        epoch, ckpt_manager = create_ckpt_and_restore(output_dir, f'{intra_idx:04d}', args, logger,
                                                      accelerator, model,
                                                      metric=eval_task.get_metrics())

        # Training loop
        start_train_time = time.time()
        if not args.eval_only:
            if epoch < args.epochs:
                logger.info(f'Start training for {args.epochs - epoch} epochs.')
            else:
                logger.info(f'Training is already completed. Skip training.')
            while epoch < args.epochs:
                # Training
                train_epoch(output_dir, f'{intra_idx:04d}/train', epoch, args, logger,
                            accelerator, model, train_loader, train_task,
                            optimizer, scheduler, do_log(args, epoch, args.epochs))

                # Evaluation
                if do_eval(args, epoch, args.epochs):
                    _, metrics = eval_epoch(output_dir, f'{intra_idx:04d}/eval', epoch, args, logger,
                                            accelerator, model, eval_loader, eval_task,
                                            do_log(args, epoch, args.epochs))
                    ckpt_manager.save(epoch, metrics)

                epoch += 1
        else:
            logger.info('Skip training. Evaluation only.')
            # Evaluation
            _, metrics = eval_epoch(output_dir, f'{intra_idx:04d}/eval', epoch, args, logger,
                                    accelerator, model, eval_loader, eval_task, True)
            ckpt_manager.save(epoch, metrics)

        train_time = time.time() - start_train_time
        logger.info(f'Training completed in: {train_time:.2f}s')

        # Write bitstream
        logger.info(f'Encoding model weights to bitstream.')
        start_enc_time = time.time()

        intra_bs = model.compress(intra_bs)
        intra_bs_len = len(intra_bs)
        bs += intra_bs_len.to_bytes(4) + intra_bs
        write_bitstream(output_dir, f'bitstream', bs)
        write_bitstream(output_dir, f'{intra_idx:04d}', intra_bs)

        enc_time = time.time() - start_enc_time
        logger.info(f'Encoding time: {enc_time:.2f}s')
        logger.info(f'Current total bitstream size: {len(bs) * 8}')
        logger.info(f'Group bitstream size: {intra_bs_len * 8}.')

        # Set model weights to zero for ensuring the correctness
        set_zero(model)

        # Decompress bitstream and evaluate
        logger.info(f'Decoding model weights from bitstream.')
        start_dec_time = time.time()
        
        intra_bs = read_bitstream(output_dir, f'{intra_idx:04d}')
        intra_bs = model.decompress(intra_bs)

        dec_time = time.time() - start_dec_time
        logger.info(f'Decoding time: {dec_time:.2f}s')
        assert len(intra_bs) == 0, f'Bitstream is not empty after decompress: {len(intra_bs)}'

        # Save decoded model
        save_model(output_dir, f'{intra_idx:04d}/decoded', None, model)

        # Evaluation
        start_eval_time = time.time()
        logger.info(f'Start evaluating the decoded model.')
        _, metrics = eval_epoch(output_dir, f'{intra_idx:04d}/decoded', None, args, logger,
                                accelerator, model, eval_loader, eval_task, True)
        eval_time = time.time() - start_eval_time
        logger.info(f'Evaluation completed in: {eval_time:.2f}s')

        # Concatenate outputs
        concate_outputs(os.path.join(output_dir, 'outputs', f'{intra_idx:04d}/decoded'),
                        os.path.join(output_dir, 'outputs', 'all', 'decoded'), args.eval_data.dataset,
                        keep_path=False)

        # Get detailed rd performance
        profile_model_rate_stat(logger, model, eval_loader, eval_task)

        # Log results
        actual_rate = (4 + intra_bs_len) * 8
        actual_bpp = actual_rate / math.prod((intra_num_frames,) + eval_task.get_video_size()[1:])
        all_num_frames.append(intra_num_frames)
        all_size.append(actual_rate)
        all_metrics.append({**{'params': num_params, 'params_codec': num_params_codec,
                                'macs': num_macs, 'macs_rate': num_macs_rate,
                                'train_time': train_time, 'eval_time': eval_time,
                                'ent_enc_time': enc_time, 'ent_dec_time': dec_time,
                                'bpp': actual_bpp, 'bpp_est': metrics['bpp']},
                            **{k: v for k, v in metrics.items() if k not in ['bpp']}})
        write_results(output_dir, f'{intra_idx:04d}', all_size[-1], all_metrics[-1])

        # End of group
        logger.info(f'Completed group encoding.\n')
        accelerator.clear()
        intra_idx += 1

    # Log results
    total_num_frames = sum(all_num_frames)
    total_size = sum(all_size)
    total_metrics = {k + '_avg': sum([m[k] * n for m, n in zip(all_metrics, all_num_frames)]) / total_num_frames \
                     for k in all_metrics[0].keys()}
    write_results(output_dir, 'all', total_size, total_metrics)

    logger.info(f'Total bitstream size: {total_size}')
    for k, v in total_metrics.items():
        logger.info(f'Average {k}: {v:.4f}')

    # Complete encoding
    total_time = time.time() - total_start_time
    logger.info(f'Encoding completed in: {total_time:.2f}s')
    logger.info(f'Output are located in: {output_dir}')
    accelerator.end_training()
    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()