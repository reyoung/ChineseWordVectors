import pyximport;

pyximport.install()
import paddle.v2.fluid as fluid
from utils.data_reader import reader_creator
import paddle.v2 as paddle

MAX_DICT_SIZE = 14898


def main(window_size, dict_size=10000, emb_size=32, num_neg_samples=10,
         batch_size=512):
    assert window_size % 2 == 1
    words = []
    for i in xrange(window_size):
        words.append(fluid.layers.data(name='word_{0}'.format(i), shape=[1],
                                       dtype='int64'))

    dict_size = min(MAX_DICT_SIZE, dict_size)
    label_word = int(window_size / 2) + 1

    for_loop = fluid.layers.ParallelDo(fluid.layers.get_places())
    with for_loop.do():
        embs = []
        for i in xrange(window_size):
            if i == label_word:
                continue

            emb = fluid.layers.embedding(input=for_loop.read_input(words[i]),
                                         size=[dict_size, emb_size],
                                         param_attr='emb.w',
                                         is_sparse=False)

            embs.append(emb)

        embs = fluid.layers.concat(input=embs, axis=1)
        loss = fluid.layers.nce(input=embs,
                                label=for_loop.read_input(words[label_word]),
                                num_total_classes=dict_size,
                                param_attr='nce.w',
                                bias_attr='nce.b',
                                num_neg_samples=num_neg_samples)
        avg_loss = fluid.layers.mean(x=loss)
        avg_loss /= num_neg_samples + 1

        for_loop.write_output(avg_loss)

    avg_loss = fluid.layers.mean(x=for_loop())
    adam = fluid.optimizer.Adagrad(learning_rate=1e-3)
    adam.minimize(loss=avg_loss)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=words, place=place)

    exe.run(fluid.default_startup_program())
    reader = paddle.batch(
        paddle.reader.buffered(
            reader_creator(window_size=window_size,
                           word_limit=dict_size - 1,
                           path="./preprocessed"), 4000), batch_size)

    for pass_id in xrange(100):
        fluid.io.save_params(exe, dirname='model_{0}'.format(pass_id))
        for batch_id, data in enumerate(reader()):
            avg_loss_np = exe.run(feed=feeder.feed(data), fetch_list=[avg_loss])
            print "Pass ID {0}, Batch ID {1}, Loss {2}".format(pass_id, batch_id, avg_loss_np[0])


if __name__ == '__main__':
    main(window_size=5, batch_size=2048)
