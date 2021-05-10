import math
import timeit
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from dataset import KnowledgeGraph
def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[(np.array([sorted_predict_score_num])*np.arange(1, 1000)/np.array([1000])).astype(int)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    TP = predict_score_matrix*real_score.T
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix=np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T

    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
    PR_dot_matrix[1,:] = -PR_dot_matrix[1,:]
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index, 0]
    accuracy = accuracy_list[max_index, 0]
    specificity = specificity_list[max_index, 0]
    recall = recall_list[max_index, 0]
    precision = precision_list[max_index, 0]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]

class TransE:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.train_op = None
        self.loss = None
        self.posloss=None
        self.negloss=None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        '''embeddings'''
        bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[kg.n_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        with tf.name_scope('link'):
            distance_pos = head_pos + relation_pos - tail_pos
            distance_neg = head_neg + relation_neg - tail_neg
        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')
            #loss = tf.reduce_sum(tf.nn.relu(margin - score_neg), name='max_margin_loss')
        self.posloss=tf.reduce_sum(score_pos)
        self.negloss=tf.reduce_sum(score_neg)
        return loss

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])
        with tf.name_scope('link'):
            distance_head_prediction = self.entity_embedding + relation - tail
            distance_tail_prediction = head + relation - self.entity_embedding
        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        allpos=0
        allneg=0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            posloss1,negloss1,batch_loss, _, summary = session.run(fetches=[self.posloss,self.negloss,self.loss, self.train_op, self.merge],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos)})
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            allpos+=posloss1
            allneg+=negloss1
            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_training_triple,
                                                                            batch_loss / len(batch_pos)), end='\r')

        print()
        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print(11111111111111111)
        print(allpos)
        print(allneg)

        print('-----Finish training-----')
        self.check_norm(session=session)

    def launch_evaluation(self, session):
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triples:
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
            print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                               n_used_eval_triple,
                                                               self.kg.n_test_triple), end='\r')
        print()
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        print('------Average------')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        print('-----Average-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                         (head_hits10_filter + tail_hits10_filter) / 2))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def check_norm(self, session):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))

    def save_embedding(self,session):
        np.save('TransE_entity_emb.npy',np.array(self.entity_embedding.eval(session=session)))
        np.save('TransE_relation_emb.npy',np.array(self.relation_embedding.eval(session=session)))

class TransD:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.train_op = None
        self.loss = None
        self.posloss=None
        self.negloss=None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        '''embeddings'''
        bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[kg.n_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            #tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            self.entity_embedding_trans = tf.get_variable(name='entity_trans',
                                                    shape=[kg.n_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            #tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding_trans = tf.get_variable(name='relation_trans',
                                                      shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            #tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
        self.build_graph()
        self.build_eval_graph()

    def calc(self, e, t, r):
        return tf.nn.l2_normalize(e + tf.reduce_sum(e * t, 1, keep_dims=True) * r, 1)
    def build_graph(self):
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
            self.entity_embedding_trans = tf.nn.l2_normalize(self.entity_embedding_trans, dim=1)
            self.relation_embedding_trans = tf.nn.l2_normalize(self.relation_embedding_trans, dim=1)
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_pos_t = tf.nn.embedding_lookup(self.entity_embedding_trans, triple_pos[:, 0])
            tail_pos_t = tf.nn.embedding_lookup(self.entity_embedding_trans, triple_pos[:, 1])
            relation_pos_t = tf.nn.embedding_lookup(self.relation_embedding_trans, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
            head_neg_t = tf.nn.embedding_lookup(self.entity_embedding_trans, triple_neg[:, 0])
            tail_neg_t = tf.nn.embedding_lookup(self.entity_embedding_trans, triple_neg[:, 1])
            relation_neg_t = tf.nn.embedding_lookup(self.relation_embedding_trans, triple_neg[:, 2])
        with tf.name_scope('link'):
            distance_pos = self.calc(head_pos,head_pos_t,relation_pos_t) + relation_pos -self.calc(tail_pos,tail_pos_t,relation_pos_t)
            distance_neg = self.calc(head_neg,head_neg_t,relation_neg_t)+relation_neg-self.calc(tail_neg,tail_neg_t,relation_neg_t)
        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')
            #loss = tf.reduce_sum(tf.nn.relu(margin - score_neg), name='max_margin_loss')
        self.posloss=tf.reduce_sum(score_pos)
        self.negloss=tf.reduce_sum(score_neg)
        return loss

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])
        with tf.name_scope('link'):
            distance_head_prediction = self.entity_embedding + relation - tail
            distance_tail_prediction = head + relation - self.entity_embedding
        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        allpos=0
        allneg=0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            posloss1,negloss1,batch_loss, _, summary = session.run(fetches=[self.posloss,self.negloss,self.loss, self.train_op, self.merge],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos)})
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            allpos+=posloss1
            allneg+=negloss1
            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_training_triple,
                                                                            batch_loss / len(batch_pos)), end='\r')

        print()
        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print(11111111111111111)
        print(allpos)
        print(allneg)

        print('-----Finish training-----')
        self.check_norm(session=session)

    def launch_evaluation(self, session):
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triples:
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
            print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                               n_used_eval_triple,
                                                               self.kg.n_test_triple), end='\r')
        print()
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        print('------Average------')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        print('-----Average-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                         (head_hits10_filter + tail_hits10_filter) / 2))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def check_norm(self, session):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))

    def save_embedding(self,session):
        np.save('TransD_entity_emb.npy',np.array(self.entity_embedding.eval(session=session)))
        np.save('TransD_relation_emb.npy',np.array(self.relation_embedding.eval(session=session)))
        np.save('TransD_entity_emb_trans.npy', np.array(self.entity_embedding_trans.eval(session=session)))
        np.save('TransD_relation_emb_trans.npy', np.array(self.relation_embedding_trans.eval(session=session)))

class TransH:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.train_op = None
        self.loss = None
        self.posloss=None
        self.negloss=None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        '''embeddings'''
        bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[kg.n_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
            self.relation_embedding_norm = tf.get_variable(name='relation_norm',
                                                      shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
        self.build_graph()
        self.build_eval_graph()

    def calc(self, e, n):
        norm = tf.nn.l2_normalize(n, 1)
        return e - tf.reduce_sum(e * norm, 1, keep_dims=True) * norm
    def build_graph(self):
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
            self.relation_embedding_norm = tf.nn.l2_normalize(self.relation_embedding_norm, dim=1)
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
            pos_r_norm=tf.nn.embedding_lookup(self.relation_embedding_norm, triple_pos[:, 2])
            neg_r_norm=tf.nn.embedding_lookup(self.relation_embedding_norm, triple_neg[:, 2])
        with tf.name_scope('link'):
            distance_pos = self.calc(head_pos,pos_r_norm)+relation_pos-self.calc(tail_pos,pos_r_norm)
            distance_neg = self.calc(head_neg,neg_r_norm)+relation_neg-self.calc(tail_neg,neg_r_norm)
        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')
            #loss = tf.reduce_sum(tf.nn.relu(margin - score_neg), name='max_margin_loss')
        self.posloss=tf.reduce_sum(score_pos)
        self.negloss=tf.reduce_sum(score_neg)
        return loss

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])
        with tf.name_scope('link'):
            distance_head_prediction = self.entity_embedding + relation - tail
            distance_tail_prediction = head + relation - self.entity_embedding
        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        allpos=0
        allneg=0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            posloss1,negloss1,batch_loss, _, summary = session.run(fetches=[self.posloss,self.negloss,self.loss, self.train_op, self.merge],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos)})
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            allpos+=posloss1
            allneg+=negloss1
            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_training_triple,
                                                                            batch_loss / len(batch_pos)), end='\r')

        print()
        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print(11111111111111111)
        print(allpos)
        print(allneg)

        print('-----Finish training-----')
        self.check_norm(session=session)

    def launch_evaluation(self, session):
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triples:
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
            print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                               n_used_eval_triple,
                                                               self.kg.n_test_triple), end='\r')
        print()
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        print('------Average------')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        print('-----Average-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                         (head_hits10_filter + tail_hits10_filter) / 2))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def check_norm(self, session):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))

    def save_embedding(self,session):
        np.save('TransH_entity_emb.npy',np.array(self.entity_embedding.eval(session=session)))
        np.save('TransH_relation_emb.npy',np.array(self.relation_embedding.eval(session=session)))
        np.save('TransH_relation_emb_norm.npy', np.array(self.relation_embedding_norm.eval(session=session)))


class DistMult:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.train_op = None
        self.loss = None
        self.posloss=None
        self.negloss=None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        '''embeddings'''
        bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[kg.n_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        with tf.name_scope('link'):
            distance_pos = head_pos * relation_pos * tail_pos
            distance_neg = head_neg * relation_neg * tail_neg
        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')
            #loss = tf.reduce_sum(tf.nn.relu(margin - score_neg), name='max_margin_loss')
        self.posloss=tf.reduce_sum(score_pos)
        self.negloss=tf.reduce_sum(score_neg)
        return loss

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])
        with tf.name_scope('link'):
            distance_head_prediction = self.entity_embedding * relation * tail
            distance_tail_prediction = head * relation * self.entity_embedding
        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        allpos=0
        allneg=0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            posloss1,negloss1,batch_loss, _, summary = session.run(fetches=[self.posloss,self.negloss,self.loss, self.train_op, self.merge],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos)})
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            allpos+=posloss1
            allneg+=negloss1
            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_training_triple,
                                                                            batch_loss / len(batch_pos)), end='\r')

        print()
        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print(11111111111111111)
        print(allpos)
        print(allneg)

        print('-----Finish training-----')
        self.check_norm(session=session)

    def launch_evaluation(self, session):
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triples:
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
            print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                               n_used_eval_triple,
                                                               self.kg.n_test_triple), end='\r')
        print()
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        print('------Average------')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        print('-----Average-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                         (head_hits10_filter + tail_hits10_filter) / 2))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def check_norm(self, session):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))

    def save_embedding(self,session):
        np.save('DistMult_entity_emb.npy',np.array(self.entity_embedding.eval(session=session)))
        np.save('DistMult_relation_emb.npy',np.array(self.relation_embedding.eval(session=session)))

class Complex:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.train_op = None
        self.loss = None
        self.posloss=None
        self.negloss=None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        '''embeddings'''
        bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope('embedding'):
            self.entity_embedding_re = tf.get_variable(name='entity_re',
                                                    shape=[kg.n_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            self.entity_embedding_im = tf.get_variable(name='entity_im',
                                                       shape=[kg.n_entity, self.embedding_dim],
                                                       initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                 maxval=bound))
            #tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding_re = tf.get_variable(name='relation_re',
                                                      shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            self.relation_embedding_im = tf.get_variable(name='relation_im',
                                                      shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            #tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
        self.build_graph()
        #self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('normalization'):
            self.entity_embedding_re = tf.nn.l2_normalize(self.entity_embedding_re, dim=1)
            self.entity_embedding_im = tf.nn.l2_normalize(self.entity_embedding_im, dim=1)
            self.relation_embedding_re = tf.nn.l2_normalize(self.relation_embedding_re, dim=1)
            self.relation_embedding_im = tf.nn.l2_normalize(self.relation_embedding_im, dim=1)
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos_re = tf.nn.embedding_lookup(self.entity_embedding_re, triple_pos[:, 0])
            tail_pos_re = tf.nn.embedding_lookup(self.entity_embedding_re, triple_pos[:, 1])
            relation_pos_re = tf.nn.embedding_lookup(self.relation_embedding_re, triple_pos[:, 2])
            head_pos_im = tf.nn.embedding_lookup(self.entity_embedding_im, triple_pos[:, 0])
            tail_pos_im = tf.nn.embedding_lookup(self.entity_embedding_im, triple_pos[:, 1])
            relation_pos_im = tf.nn.embedding_lookup(self.relation_embedding_im, triple_pos[:, 2])
            head_neg_re = tf.nn.embedding_lookup(self.entity_embedding_re, triple_neg[:, 0])
            tail_neg_re = tf.nn.embedding_lookup(self.entity_embedding_re, triple_neg[:, 1])
            relation_neg_re = tf.nn.embedding_lookup(self.relation_embedding_re, triple_neg[:, 2])
            head_neg_im = tf.nn.embedding_lookup(self.entity_embedding_im, triple_neg[:, 0])
            tail_neg_im = tf.nn.embedding_lookup(self.entity_embedding_im, triple_neg[:, 1])
            relation_neg_im = tf.nn.embedding_lookup(self.relation_embedding_im, triple_neg[:, 2])
        with tf.name_scope('link'):
            distance_pos = head_pos_re*tail_pos_re*relation_pos_re+head_pos_im*tail_pos_im*relation_pos_re+head_pos_re*tail_pos_im*relation_pos_im-head_pos_im*tail_pos_re*relation_pos_im
            distance_neg = head_neg_re*tail_neg_re*relation_neg_re+head_neg_im*tail_neg_im*relation_neg_re+head_neg_re*tail_neg_im*relation_neg_im-head_neg_im*tail_neg_re*relation_neg_im
        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')
            #loss = tf.reduce_sum(tf.nn.relu(margin - score_neg), name='max_margin_loss')
        self.posloss=tf.reduce_sum(score_pos)
        self.negloss=tf.reduce_sum(score_neg)
        return loss

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head_re = tf.nn.embedding_lookup(self.entity_embedding_re, eval_triple[0])
            tail_re = tf.nn.embedding_lookup(self.entity_embedding_re, eval_triple[1])
            relation_re = tf.nn.embedding_lookup(self.relation_embedding_re, eval_triple[2])
            head_im = tf.nn.embedding_lookup(self.entity_embedding_im, eval_triple[0])
            tail_im = tf.nn.embedding_lookup(self.entity_embedding_im, eval_triple[1])
            relation_im = tf.nn.embedding_lookup(self.relation_embedding_im, eval_triple[2])
        with tf.name_scope('link'):
            distance_head_prediction = self.entity_embedding * relation * tail
            distance_tail_prediction = head * relation * self.entity_embedding
        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        allpos=0
        allneg=0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            posloss1,negloss1,batch_loss, _, summary = session.run(fetches=[self.posloss,self.negloss,self.loss, self.train_op, self.merge],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos)})
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            allpos+=posloss1
            allneg+=negloss1
            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_training_triple,
                                                                            batch_loss / len(batch_pos)), end='\r')

        print()
        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print(11111111111111111)
        print(allpos)
        print(allneg)

        print('-----Finish training-----')
        self.check_norm(session=session)

    def launch_evaluation(self, session):
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triples:
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
            print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                               n_used_eval_triple,
                                                               self.kg.n_test_triple), end='\r')
        print()
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        print('------Average------')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        print('-----Average-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                         (head_hits10_filter + tail_hits10_filter) / 2))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def check_norm(self, session):
        print('-----Check norm-----')
        #entity_embedding = self.entity_embedding.eval(session=session)
        #relation_embedding = self.relation_embedding.eval(session=session)
        #entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        #relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        #print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))

    def save_embedding(self,session):
        np.save('Complex_entity_emb_re.npy',np.array(self.entity_embedding_re.eval(session=session)))
        np.save('Complex_relation_emb_re.npy',np.array(self.relation_embedding_re.eval(session=session)))
        np.save('Complex_entity_emb_im.npy', np.array(self.entity_embedding_im.eval(session=session)))
        np.save('Complex_relation_emb_im.npy', np.array(self.relation_embedding_im.eval(session=session)))

class Rescal:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.train_op = None
        self.loss = None
        self.posloss=None
        self.negloss=None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        '''embeddings'''
        bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[kg.n_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))

            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[kg.n_relation,self.embedding_dim,self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))

            #tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
        self.build_graph()
        #self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)

            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)

        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])

            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])

        with tf.name_scope('link'):
            distance_pos = head_pos*tf.reshape(tf.matmul(relation_pos,tf.reshape(tail_pos,(-1,self.embedding_dim,1))),(-1,self.embedding_dim))
            distance_neg = head_neg*tf.reshape(tf.matmul(relation_neg,tf.reshape(tail_neg,(-1,self.embedding_dim,1))),(-1,self.embedding_dim))
        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')
            #loss = tf.reduce_sum(tf.nn.relu(margin - score_neg), name='max_margin_loss')
        self.posloss=tf.reduce_sum(score_pos)
        self.negloss=tf.reduce_sum(score_neg)
        return loss

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head_re = tf.nn.embedding_lookup(self.entity_embedding_re, eval_triple[0])
            tail_re = tf.nn.embedding_lookup(self.entity_embedding_re, eval_triple[1])
            relation_re = tf.nn.embedding_lookup(self.relation_embedding_re, eval_triple[2])
            head_im = tf.nn.embedding_lookup(self.entity_embedding_im, eval_triple[0])
            tail_im = tf.nn.embedding_lookup(self.entity_embedding_im, eval_triple[1])
            relation_im = tf.nn.embedding_lookup(self.relation_embedding_im, eval_triple[2])
        with tf.name_scope('link'):
            distance_head_prediction = self.entity_embedding * relation * tail
            distance_tail_prediction = head * relation * self.entity_embedding
        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        allpos=0
        allneg=0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            posloss1,negloss1,batch_loss, _, summary = session.run(fetches=[self.posloss,self.negloss,self.loss, self.train_op, self.merge],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos)})
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            allpos+=posloss1
            allneg+=negloss1
            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_training_triple,
                                                                            batch_loss / len(batch_pos)), end='\r')

        print()
        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print(11111111111111111)
        print(allpos)
        print(allneg)

        print('-----Finish training-----')
        self.check_norm(session=session)

    def launch_evaluation(self, session):
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triples:
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
            print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                               n_used_eval_triple,
                                                               self.kg.n_test_triple), end='\r')
        print()
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        print('------Average------')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        print('-----Average-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                         (head_hits10_filter + tail_hits10_filter) / 2))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def check_norm(self, session):
        print('-----Check norm-----')
        #entity_embedding = self.entity_embedding.eval(session=session)
        #relation_embedding = self.relation_embedding.eval(session=session)
        #entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        #relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        #print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))

    def save_embedding(self,session):
        np.save('rescal_entity_emb.npy',np.array(self.entity_embedding.eval(session=session)))
        np.save('rescal_relation_emb.npy',np.array(self.relation_embedding.eval(session=session)))

class NAM:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.truey=tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.train_op = None
        self.loss = None
        self.posloss=None
        self.negloss=None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        self.outputpre=None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[None,3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        self.weights = {
            'l1': tf.get_variable("Wl1", shape=[300, 200], initializer=tf.contrib.layers.xavier_initializer()),
            'l2': tf.get_variable("Wl2", shape=[200, 200],
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'l3': tf.get_variable("Wl3", shape=[200, 200],
                                  initializer=tf.contrib.layers.xavier_initializer())
        }

        self.biases = {
            'l1': tf.get_variable("bl1", shape=[200], initializer=tf.contrib.layers.xavier_initializer()),
            'l2': tf.get_variable("bl2", shape=[200], initializer=tf.contrib.layers.xavier_initializer()),
            'l3': tf.get_variable("bl3", shape=[200], initializer=tf.contrib.layers.xavier_initializer())
        }
        '''embeddings'''
        bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[kg.n_entity, 200],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            # tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[kg.n_relation, 100],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.outputpre = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        with tf.name_scope('link'):
            z0 = tf.concat([relation_pos, head_pos], axis=1)
            l1 = tf.matmul(z0, self.weights['l1']) + self.biases['l1']
            z1 = tf.nn.relu(l1)

            l2 = tf.matmul(z1, self.weights['l2']) + self.biases['l2']
            z2 = tf.nn.relu(l2)

            l3 = tf.matmul(z2, self.weights['l3']) + self.biases['l3']
            z3 = tf.nn.relu(l3)

            m = tf.multiply(z3, tail_pos)
            dot = tf.reduce_sum(m, axis=1)
            output = tf.nn.sigmoid(dot)  # Output shape should be batch_size*1
            distance_pos = output
            z0 = tf.concat([relation_neg, head_neg], axis=1)
            l1 = tf.matmul(z0, self.weights['l1']) + self.biases['l1']
            z1 = tf.nn.relu(l1)

            l2 = tf.matmul(z1, self.weights['l2']) + self.biases['l2']
            z2 = tf.nn.relu(l2)

            l3 = tf.matmul(z2, self.weights['l3']) + self.biases['l3']
            z3 = tf.nn.relu(l3)

            m = tf.multiply(z3, tail_neg)
            dot = tf.reduce_sum(m, axis=1)
            output = tf.nn.sigmoid(dot)  # Output shape should be batch_size*1
            distance_neg = output
        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        distance_neg=tf.reshape(distance_neg,[-1,1])
        distance_pos = tf.reshape(distance_pos, [-1, 1])
        pre=tf.concat([distance_pos, distance_neg], axis=0)
        #self.outputpre=pre
        print(pre.shape)
        cost = tf.reduce_mean(-(tf.multiply(self.truey, tf.log(tf.clip_by_value(pre,1e-8,1.0))) + tf.multiply(1 - self.truey, tf.log(tf.clip_by_value(1 - pre,1e-8,1.0)))))

        return cost

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[:,0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[:,1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[:,2])
        with tf.name_scope('link'):
            # head=tf.reshape(head,[1,-1])
            # tail = tf.reshape(tail, [1, -1])
            # tail = tf.reshape(tail, [1, -1])
            z0 = tf.concat([relation, head], axis=1)
            l1 = tf.matmul(z0, self.weights['l1']) + self.biases['l1']
            z1 = tf.nn.relu(l1)

            l2 = tf.matmul(z1, self.weights['l2']) + self.biases['l2']
            z2 = tf.nn.relu(l2)

            l3 = tf.matmul(z2, self.weights['l3']) + self.biases['l3']
            z3 = tf.nn.relu(l3)

            m = tf.multiply(z3, tail)
            dot = tf.reduce_sum(m, axis=1)
            output = tf.nn.sigmoid(dot)  # Output shape should be batch_size*1
            print(111111111111111111)
            print(output.shape)

        return output

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        allpos=0
        allneg=0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            yyy=np.zeros((2*len(batch_pos),1))
            yyy[0:len(batch_pos)]=1

            batch_loss, _ = session.run(fetches=[self.loss, self.train_op],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos),
                                                            self.truey:yyy})
            #summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss

            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_training_triple,
                                                                            batch_loss / len(batch_pos)), end='\r')

        print()
        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))


        print('-----Finish training-----')
        #self.check_norm(session=session)

    def launch_evaluation(self, session):
        with open('data//' + 'humancell'+ '//entities.txt') as f:
            entity2id = {}
            for line in f:
                eid, entity1 = line.strip().split('\t')
                entity2id[eid] = int(entity1)

        b = np.loadtxt('data//' + 'humancell' + '//valid.txt', dtype=str, delimiter='\t')
        c = np.loadtxt('data//' + 'humancell' + '//neg_val.txt', dtype=str, delimiter='\t')
        tri=np.zeros((b.shape[0]+c.shape[0],3))
        iall=0
        for i in range(b.shape[0]):
            tri[i][0]=entity2id[b[i][0]]
            tri[i][1] = entity2id[b[i][1]]
            tri[i][2] = 2
            iall=i+1
        for i in range(c.shape[0]):
            tri[iall+i][0] = entity2id[c[i][0]]
            tri[iall+i][1] = entity2id[c[i][1]]
            tri[iall+i][2] = 2
        print(tri)
        tri=list(tri)
        print(len(tri))
        #for eval_triple in tri:
        prediction= session.run(fetches=[self.outputpre],feed_dict={self.eval_triple: tri})
        prediction=np.array(prediction)
        prediction=prediction.T
        label=np.zeros((prediction.shape[0]))
        label[0:prediction.shape[0]//2]=1

        print(prediction.shape)
        print(label.shape)
        aaa=get_metrics(np.mat(label),np.mat(prediction.T))
        print(aaa)
        with open('NAMresult.txt', 'a') as f:

            f.write(str(aaa)+'\n')

            # print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
            #                                                    n_used_eval_triple,
            #                                                    self.kg.n_test_triple), end='\r')
        print()


    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def check_norm(self, session):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))

    def save_embedding(self,session):
        np.save('NAM_entity_emb.npy',np.array(self.entity_embedding.eval(session=session)))
        np.save('NAM_relation_emb.npy',np.array(self.relation_embedding.eval(session=session)))

class MLP:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.truey=tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.train_op = None
        self.loss = None
        self.posloss=None
        self.negloss=None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        self.outputpre=None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[None,3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        self.weights = {
            'l1': tf.get_variable("Wl1", shape=[200, 200], initializer=tf.contrib.layers.xavier_initializer()),
            'l2': tf.get_variable("Wl2", shape=[200, 200],
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'l3': tf.get_variable("Wl3", shape=[200, 200],
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'l4': tf.get_variable("Wl4", shape=[200, 1], initializer=tf.contrib.layers.xavier_initializer())
        }

        # self.biases = {
        #     'l1': tf.get_variable("bl1", shape=[200], initializer=tf.contrib.layers.xavier_initializer()),
        #     'l2': tf.get_variable("bl2", shape=[200], initializer=tf.contrib.layers.xavier_initializer()),
        #     'l3': tf.get_variable("bl3", shape=[200], initializer=tf.contrib.layers.xavier_initializer())
        # }
        '''embeddings'''
        bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[kg.n_entity, 200],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            # tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[kg.n_relation, 200],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.outputpre = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        with tf.name_scope('link'):

            x1 = tf.matmul(head_pos, self.weights['l1'])
            x2 = tf.matmul(tail_pos, self.weights['l2'])
            x3 = tf.matmul(relation_pos, self.weights['l3'])
            xall=x1+x2+x3
            xall=tf.nn.tanh(xall)
            output=tf.matmul(xall,self.weights['l4'])
            output=tf.reshape(output,[-1,1])
            distance_pos = output

            x1 = tf.matmul(head_neg, self.weights['l1'])
            x2 = tf.matmul(tail_neg, self.weights['l2'])
            x3 = tf.matmul(relation_neg, self.weights['l3'])
            xall = x1 + x2 + x3
            xall = tf.nn.tanh(xall)
            output = tf.matmul(xall, self.weights['l4'])
            output = tf.reshape(output, [-1, 1])
            distance_neg = output

        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        distance_neg=tf.reshape(distance_neg,[-1,1])
        distance_pos = tf.reshape(distance_pos, [-1, 1])
        pre=tf.concat([distance_pos, distance_neg], axis=0)
        #self.outputpre=pre
        print(pre.shape)
        cost = tf.reduce_mean(-(tf.multiply(self.truey, tf.log(tf.clip_by_value(pre,1e-8,1.0))) + tf.multiply(1 - self.truey, tf.log(tf.clip_by_value(1 - pre,1e-8,1.0)))))

        return cost

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[:,0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[:,1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[:,2])
        with tf.name_scope('link'):
            x1 = tf.matmul(head, self.weights['l1'])
            x2 = tf.matmul(tail, self.weights['l2'])
            x3 = tf.matmul(relation, self.weights['l3'])
            xall = x1 + x2 + x3
            xall = tf.nn.tanh(xall)
            output = tf.matmul(xall, self.weights['l4'])
            output=tf.reshape(output,[-1,1])
            #output = tf.reshape(output, [-1, 1])


        return output

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        allpos=0
        allneg=0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            yyy=np.zeros((2*len(batch_pos),1))
            yyy[0:len(batch_pos)]=1

            batch_loss, _ = session.run(fetches=[self.loss, self.train_op],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos),
                                                            self.truey:yyy})
            #summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss

            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_training_triple,
                                                                            batch_loss / len(batch_pos)), end='\r')

        print()
        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))


        print('-----Finish training-----')
        #self.check_norm(session=session)

    def launch_evaluation(self, session):
        with open('data//' + 'humancell'+ '//entities.txt') as f:
            entity2id = {}
            for line in f:
                eid, entity1 = line.strip().split('\t')
                entity2id[eid] = int(entity1)

        b = np.loadtxt('data//' + 'humancell' + '//valid.txt', dtype=str, delimiter='\t')
        c = np.loadtxt('data//' + 'humancell' + '//neg_val.txt', dtype=str, delimiter='\t')
        tri=np.zeros((b.shape[0]+c.shape[0],3))
        iall=0
        for i in range(b.shape[0]):
            tri[i][0]=entity2id[b[i][0]]
            tri[i][1] = entity2id[b[i][1]]
            tri[i][2] = 2
            iall=i+1
        for i in range(c.shape[0]):
            tri[iall+i][0] = entity2id[c[i][0]]
            tri[iall+i][1] = entity2id[c[i][1]]
            tri[iall+i][2] = 2
        print(tri)
        tri=list(tri)
        print(len(tri))
        #for eval_triple in tri:
        prediction= session.run(fetches=[self.outputpre],feed_dict={self.eval_triple: tri})
        prediction=np.array(prediction)
        prediction=np.reshape(prediction,(-1,1))
        print(prediction.shape)
        #prediction=prediction.T
        label=np.zeros((prediction.shape[0]))
        label[0:prediction.shape[0]//2]=1


        print(label.shape)
        aaa=get_metrics(np.mat(label),np.mat(prediction.T))
        print(aaa)
        with open('MLPresult.txt', 'a') as f:

            f.write(str(aaa)+'\n')

            # print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
            #                                                    n_used_eval_triple,
            #                                                    self.kg.n_test_triple), end='\r')
        print()


    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def check_norm(self, session):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))

    def save_embedding(self,session):
        np.save('MLP_entity_emb.npy',np.array(self.entity_embedding.eval(session=session)))
        np.save('MLP_relation_emb.npy',np.array(self.relation_embedding.eval(session=session)))

class BPRMF:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.truey=tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.train_op = None
        self.loss = None
        self.posloss=None
        self.negloss=None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        self.outputpre=None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[None,3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        # self.weights = {
        #     'l1': tf.get_variable("Wl1", shape=[200, 200], initializer=tf.contrib.layers.xavier_initializer()),
        #     'l2': tf.get_variable("Wl2", shape=[200, 200],
        #                           initializer=tf.contrib.layers.xavier_initializer()),
        #     'l3': tf.get_variable("Wl3", shape=[200, 200],
        #                           initializer=tf.contrib.layers.xavier_initializer()),
        #     'l4': tf.get_variable("Wl4", shape=[200, 1], initializer=tf.contrib.layers.xavier_initializer())
        # }

        # self.biases = {
        #     'l1': tf.get_variable("bl1", shape=[200], initializer=tf.contrib.layers.xavier_initializer()),
        #     'l2': tf.get_variable("bl2", shape=[200], initializer=tf.contrib.layers.xavier_initializer()),
        #     'l3': tf.get_variable("bl3", shape=[200], initializer=tf.contrib.layers.xavier_initializer())
        # }
        '''embeddings'''
        bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[kg.n_entity, 200],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            # tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[kg.n_relation, 200],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.triple_pos, self.triple_neg)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.outputpre = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        with tf.name_scope('link'):

            output=head_pos*tail_pos
            output=tf.reduce_sum(output,axis=1)
            distance_pos=output

            output=head_neg*tail_neg
            output=tf.reduce_sum(output,axis=1)
            distance_neg = output

        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg,triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])

        regularizer = tf.nn.l2_loss(head_pos) + tf.nn.l2_loss(head_neg)+tf.nn.l2_loss(tail_pos) + tf.nn.l2_loss(tail_neg)

        maxi = tf.log(tf.nn.sigmoid(distance_pos - distance_neg))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        kge_loss = tf.constant(0.0, tf.float32, [1])
        reg_loss = 1e-5 * regularizer
        allloss=mf_loss+kge_loss+reg_loss
        #reg_loss = self.regs[0] * regularizer

        return allloss#, reg_loss
        #return cost

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[:,0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[:,1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[:,2])
        with tf.name_scope('link'):
            output=head*tail
            output=tf.reduce_sum(output,axis=1)
            #output = tf.reshape(output, [-1, 1])


        return output

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        allpos=0
        allneg=0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            yyy=np.zeros((2*len(batch_pos),1))
            yyy[0:len(batch_pos)]=1

            batch_loss, _ = session.run(fetches=[self.loss, self.train_op],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos),
                                                            self.truey:yyy})
            #summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss

            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_training_triple,
                                                                            float(batch_loss) / len(batch_pos)), end='\r')

        print()
        print('epoch loss: {:.3f}'.format(float(epoch_loss)))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))


        print('-----Finish training-----')
        #self.check_norm(session=session)

    def launch_evaluation(self, session):
        with open('data//' + 'humancell'+ '//entities.txt') as f:
            entity2id = {}
            for line in f:
                eid, entity1 = line.strip().split('\t')
                entity2id[eid] = int(entity1)

        b = np.loadtxt('data//' + 'humancell' + '//valid.txt', dtype=str, delimiter='\t')
        c = np.loadtxt('data//' + 'humancell' + '//neg_val.txt', dtype=str, delimiter='\t')
        tri=np.zeros((b.shape[0]+c.shape[0],3))
        iall=0
        for i in range(b.shape[0]):
            tri[i][0]=entity2id[b[i][0]]
            tri[i][1] = entity2id[b[i][1]]
            tri[i][2] = 2
            iall=i+1
        for i in range(c.shape[0]):
            tri[iall+i][0] = entity2id[c[i][0]]
            tri[iall+i][1] = entity2id[c[i][1]]
            tri[iall+i][2] = 2
        print(tri)
        tri=list(tri)
        print(len(tri))
        #for eval_triple in tri:
        prediction= session.run(fetches=[self.outputpre],feed_dict={self.eval_triple: tri})
        prediction=np.array(prediction)
        prediction=np.reshape(prediction,(-1,1))
        print(prediction.shape)
        #prediction=prediction.T
        label=np.zeros((prediction.shape[0]))
        label[0:prediction.shape[0]//2]=1


        print(label.shape)
        aaa=get_metrics(np.mat(label),np.mat(prediction.T))
        print(aaa)
        with open('BPRMFresult.txt', 'a') as f:

            f.write(str(aaa)+'\n')

            # print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
            #                                                    n_used_eval_triple,
            #                                                    self.kg.n_test_triple), end='\r')
        print()


    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def check_norm(self, session):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))

    def save_embedding(self,session):
        np.save('MLP_entity_emb.npy',np.array(self.entity_embedding.eval(session=session)))
        np.save('MLP_relation_emb.npy',np.array(self.relation_embedding.eval(session=session)))

class KGAT:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.truey = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.train_op = None
        self.loss = None
        self.posloss = None
        self.negloss = None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        self.outputpre = None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        # self.weights = {
        #     'l1': tf.get_variable("Wl1", shape=[200, 200], initializer=tf.contrib.layers.xavier_initializer()),
        #     'l2': tf.get_variable("Wl2", shape=[200, 200],
        #                           initializer=tf.contrib.layers.xavier_initializer()),
        #     'l3': tf.get_variable("Wl3", shape=[200, 200],
        #                           initializer=tf.contrib.layers.xavier_initializer()),
        #     'l4': tf.get_variable("Wl4", shape=[200, 1], initializer=tf.contrib.layers.xavier_initializer())
        # }

        # self.biases = {
        #     'l1': tf.get_variable("bl1", shape=[200], initializer=tf.contrib.layers.xavier_initializer()),
        #     'l2': tf.get_variable("bl2", shape=[200], initializer=tf.contrib.layers.xavier_initializer()),
        #     'l3': tf.get_variable("bl3", shape=[200], initializer=tf.contrib.layers.xavier_initializer())
        # }
        '''embeddings'''
        initializer = tf.contrib.layers.xavier_initializer()
        bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope('embedding'):
            self.user_embedding = tf.Variable(initializer([267, 100]), name='user_embed')
            self.item_embedding = tf.Variable(initializer([570, 100]), name='item_embed')
            # tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation1',
                                                      shape=[kg.n_relation, 200],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            #tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('normalization'):
             self.user_embedding = tf.nn.l2_normalize(self.user_embedding, dim=1)
             self.item_embedding = tf.nn.l2_normalize(self.item_embedding, dim=1)
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.triple_pos, self.triple_neg)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.outputpre = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.user_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.item_embedding, triple_pos[:, 1])
            #relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.user_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.item_embedding, triple_neg[:, 1])
            #relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        with tf.name_scope('link'):
            output = head_pos * tail_pos
            output = tf.reduce_sum(output, axis=1)
            distance_pos = output

            output = head_neg * tail_neg
            output = tf.reduce_sum(output, axis=1)
            distance_neg = output

        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.user_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.item_embedding, triple_pos[:, 1])
            #relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.user_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.item_embedding, triple_neg[:, 1])
            #relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])

        # regularizer = tf.nn.l2_loss(head_pos) + tf.nn.l2_loss(head_neg) + tf.nn.l2_loss(tail_pos) + tf.nn.l2_loss(
        #     tail_neg)

        #maxi = tf.log(tf.nn.sigmoid(distance_pos - distance_neg))
        #
        self.distance_pos_loss=tf.reduce_mean(distance_pos)

        self.distance_neg_loss=tf.reduce_mean(distance_neg)
        allloss = tf.reduce_mean(tf.nn.softplus(-(distance_pos-distance_neg)))
        # # kge_loss = tf.constant(0.0, tf.float32, [1])
        # # reg_loss = 1e-5 * regularizer
        # allloss = mf_loss #+ kge_loss + reg_loss
        # # reg_loss = self.regs[0] * regularizer
        #self.distance_pos=tf.reduce_mean(distance_pos)
        #self.distance_neg=tf.reduce_mean(distance_neg)
        #allloss=-(distance_pos-distance_neg)/2
        return allloss  # , reg_loss#
        # return cost

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.user_embedding, eval_triple[:, 0])
            tail = tf.nn.embedding_lookup(self.item_embedding, eval_triple[:, 1])
            #relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[:, 2])
        with tf.name_scope('link'):
            output = head * tail
            output = tf.reduce_sum(output, axis=1)
            # output = tf.reshape(output, [-1, 1])

        return output

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        epoch_loss1=0
        n_used_triple = 0
        allpos = 0
        allneg = 0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            yyy = np.zeros((2 * len(batch_pos), 1))
            yyy[0:len(batch_pos)] = 1

            batch_loss,batch_loss1,_ = session.run(fetches=[self.distance_pos_loss,self.distance_neg_loss, self.train_op],
                                        feed_dict={self.triple_pos: batch_pos,
                                                   self.triple_neg: batch_neg,
                                                   self.margin: [self.margin_value] * len(batch_pos),
                                                   self.truey: yyy})
            # summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            epoch_loss1+=batch_loss1

            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_training_triple,
                                                                            float(batch_loss) / len(batch_pos)), end
                  ='\r')

        print()
        print('epoch loss: {:.3f}'.format(float(epoch_loss)))
        print('epoch neg loss: {:.3f}'.format(float(epoch_loss1)))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))

        print('-----Finish training-----')
        # self.check_norm(session=session)

    def launch_evaluation(self, session):
        with open('data//' + 'humancell' + '//drugnegentities.txt') as f:
            entity2id = {}
            for line in f:
                eid, entity1 = line.strip().split('\t')
                entity2id[eid] = int(entity1)

        with open('data//' + 'humancell' + '//diseasenegentities.txt') as f:
            entity2id1 = {}
            for line in f:
                eid, entity1 = line.strip().split('\t')
                entity2id1[eid] = int(entity1)

        b = np.loadtxt('data//' + 'humancell' + '//valid.txt', dtype=str, delimiter='\t')
        c = np.loadtxt('data//' + 'humancell' + '//neg_val.txt', dtype=str, delimiter='\t')
        tri = np.zeros((b.shape[0] + c.shape[0], 3))
        iall = 0
        for i in range(b.shape[0]):
            tri[i][0] = entity2id[b[i][0]]
            tri[i][1] = entity2id1[b[i][1]]
            tri[i][2] = 2
            iall = i + 1
        for i in range(c.shape[0]):
            tri[iall + i][0] = entity2id[c[i][0]]
            tri[iall + i][1] = entity2id1[c[i][1]]
            tri[iall + i][2] = 2
        print(tri)
        tri = list(tri)
        print(len(tri))
        # for eval_triple in tri:
        prediction = session.run(fetches=[self.outputpre], feed_dict={self.eval_triple: tri})
        prediction = np.array(prediction)
        prediction = np.reshape(prediction, (-1, 1))
        print(prediction.shape)
        prediction=np.reshape(prediction,(-1,))
        # prediction=prediction.T
        label = np.zeros((prediction.shape[0]))
        label[0:prediction.shape[0] // 2] = 1
        print(prediction[0:100])
        print(prediction[-100:-1])
        print(label.shape)
        aaa = get_metrics(np.mat(label), np.mat(prediction.T))
        print(aaa)
        with open('BPRMFresult.txt', 'a') as f:

            f.write(str(aaa) + '\n')

            # print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
            #                                                    n_used_eval_triple,
            #                                                    self.kg.n_test_triple), end='\r')
        print()

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    # def check_norm(self, session):
    #     print('-----Check norm-----')
    #     entity_embedding = self.entity_embedding.eval(session=session)
    #     relation_embedding = self.relation_embedding.eval(session=session)
    #     entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
    #     relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
    #     print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))
    #
    # def save_embedding(self, session):
    #     np.save('MLP_entity_emb.npy', np.array(self.entity_embedding.eval(session=session)))
    #     np.save('MLP_relation_emb.npy', np.array(self.relation_embedding.eval(session=session)))
