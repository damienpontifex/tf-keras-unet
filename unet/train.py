#!/usr/bin/env python3

import os
from argparse import ArgumentParser
import tensorflow as tf

import model
import data

def main():
    parser = ArgumentParser()
    parser.add_argument('--data-dir', help='Location where TFRecord files are')
    parser.add_argument('--model-dir', help='Location where model will be saved')
    parser.add_argument('--max-steps', default=10000)

    args, _ = parser.parse_known_args()

    # Ensure our model directory exists
    tf.gfile.MakeDirs(args.model_dir)


    config = tf.estimator.RunConfig(
        train_distribute=tf.contrib.distribute.MirroredStrategy())

    estimator = tf.estimator.Estimator(
        model_fn=model.unet_model, 
        model_dir=args.model_dir,
        config=config,
        params={
            'learning_rate': 1e-4
        }
    )

    tf.logging.set_verbosity(tf.logging.INFO)
    
    train_input_fn = lambda: data.make_dataset(os.path.join(args.data_dir, 'train-*.tfrecord'), batch_size=2, shuffle=True)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=args.max_steps) #, hooks=[debug_hook])

    val_input_fn = lambda: data.make_dataset(os.path.join(args.data_dir, 'validation-0.tfrecord'), batch_size=2, shuffle=False)
    eval_spec = tf.estimator.EvalSpec(val_input_fn, steps=20)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    main()
