import gzip
import os
import sys
import math
from utils.data_reader import reader_creator

import paddle.v2 as paddle

MAX_WORDS = 1705935


def cbow_main(cost_config, window_size=5, prefix="./output/cbow_softmax/",
              cpu_num=3,
              word_dict_limit=20000, emb_size=32):
    assert word_dict_limit < MAX_WORDS
    assert window_size % 2 == 1
    paddle.init(use_gpu=False, trainer_count=cpu_num)
    words = []

    word_limit = word_dict_limit + 2

    for i in xrange(window_size):
        words.append(paddle.layer.data(name='word_%d' % i,
                                       type=paddle.data_type.integer_value(
                                           word_limit)))

    embs = []
    for w in words[:window_size / 2] + words[-window_size / 2 + 1:]:
        embs.append(
            paddle.layer.embedding(input=w, size=emb_size, param_attr=
            paddle.attr.Param(name='emb', sparse_update=True)))

    with paddle.layer.mixed(size=emb_size) as sum_emb:
        for emb in embs:
            sum_emb += paddle.layer.identity_projection(input=emb)

    label = words[window_size / 2]

    cost = cost_config(sum_emb, label, word_limit)

    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.RMSProp(
        learning_rate=1e-3)
    trainer = paddle.trainer.SGD(cost, parameters, adam_optimizer)

    counter = [0]
    total_cost = [0.0]

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            total_cost[0] += event.cost
            counter[0] += 1

            sys.stdout.write('.')
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, AvgCost %f" % (
                    event.pass_id, event.batch_id, total_cost[0] / counter[0])
            if event.batch_id % 1000000 == 0:
                with gzip.open(os.path.join(prefix, "model_%d_%d.tar.gz" % (
                        event.pass_id,
                        event.batch_id)),
                               'w') as f:
                    parameters.to_tar(f)

        if isinstance(event, paddle.event.EndPass):
            print "Pass %d" % event.pass_id
            with gzip.open(
                    os.path.join(prefix, "model_%d.tar.gz" % event.pass_id,
                                 'w')) as f:
                parameters.to_tar(f)

    trainer.train(
        paddle.batch(
            paddle.reader.buffered(
                reader_creator(window_size=window_size,
                               word_limit=word_dict_limit,
                               path="./preprocessed"), 16 * cpu_num * 4000),
            96 * cpu_num),
        num_passes=1,
        event_handler=event_handler,
        feeding=[w.name for w in words])


def softmax_cost(sum_emb, label, word_limit):
    prediction = paddle.layer.fc(input=sum_emb,
                                 size=word_limit,
                                 act=paddle.activation.Softmax())

    return paddle.layer.classification_cost(input=prediction, label=label)


def hsigmoid_cost(sum_emb, label, word_limit):
    return paddle.layer.hsigmoid(input=sum_emb, label=label,
                                 num_classes=word_limit)


def nce_cost(sum_emb, label, word_limit):
    word_count = []
    s = 0
    with open('word_dict', 'r') as f:
        for i, line in enumerate(f):
            if i == word_limit - 2: break
            cnt = int(line.split()[0])
            s += cnt
            word_count.append(cnt)
        word_count.append(s / (word_limit - 2))
        word_count.append(s / (word_limit - 2))
        s += word_count[-1] * 2
    assert len(word_count) == word_limit
    word_count = [float(x) / s for x in word_count]
    return paddle.layer.nce(input=sum_emb, label=label, num_classes=word_limit,
                            neg_distribution=word_count)


if __name__ == '__main__':
    cbow_main(nce_cost, window_size=11)
