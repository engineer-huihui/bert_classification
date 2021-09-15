# -*- coding: UTF-8 -*-  

from extractFeature import *
import numpy as np
from arguments import arg_dic
class classification:
	
	#初始化操作
	def __init__(self):
		self.label_list = []
		self.num_labels = 0
		self.graph_cls = tf.Graph()
		self.probabilities = np.zeros(self.num_labels)
		self.sess = tf.Session()
		self.tokenizer = tokenization.FullTokenizer(vocab_file=arg_dic['vocab_file'],do_lower_case=arg_dic['do_lower_case'])
		self._load_graph()	
	#加载图结构
	def _load_graph(self):
		label_path="./input/label.txt"
		tf_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
		tf_config.gpu_options.allow_growth = True
		tokenization.validate_case_matches_checkpoint(arg_dic['do_lower_case'], arg_dic['init_checkpoint'])
		config = modeling.BertConfig.from_json_file(arg_dic['bert_config_file'])
		self.label_list = [label.strip() for label in open(label_path,'r',encoding='utf-8').readlines()]
		self.num_labels = len(self.label_list)
		graph_cls = tf.get_default_graph()
		self.sess = tf.Session(graph=graph_cls,config=tf_config)
		with self.sess.as_default():
			with graph_cls.as_default():
				
				self.input_ids_p=tf.placeholder(tf.int32,[1,arg_dic['max_seq_length']],name='input_ids')
				self.imput_mask_p=tf.placeholder(tf.int32,[1,arg_dic['max_seq_length']],name='input_mask')
				self.label_ids_p = tf.placeholder(tf.int32, [1],name='lable_ids')
				self.segment_ids_p = tf.placeholder(tf.int32, [arg_dic['max_seq_length']],name='segment_ids')
				log_probs,per_example_loss,self.logits,self.probabilities=create_model(config, False, self.input_ids_p, self.imput_mask_p, self.segment_ids_p,self.label_ids_p, self.num_labels,False)
				saver=tf.train.Saver()
				saver.restore(self.sess,"output/model.ckpt-5000")

	#模型推理
	def predictClass(self,text):
		example= InputExample(guid=0, text_a=text,text_b=None,label=self.label_list[0])
		feature = convert_single_example(0, example, self.label_list, arg_dic['max_seq_length'], self.tokenizer)
		input_ids = np.reshape([feature.input_ids], (1, arg_dic['max_seq_length']))
		input_mask = np.reshape([feature.input_mask], (1, arg_dic['max_seq_length']))
		segment_ids = np.reshape([feature.segment_ids], (arg_dic['max_seq_length']))
		lable_ids = [feature.label_id]
		feed_dict = {self.input_ids_p: input_ids, self.imput_mask_p: input_mask,self.label_ids_p:lable_ids,self.segment_ids_p:segment_ids}
		probabilities_10 = self.sess.run(self.probabilities, feed_dict)
		label_prob_dict = []
		for i, label in enumerate(self.label_list):
			if probabilities_10[0][i]>0.5:
				label_prob_dict.append(label)
		print(label_prob_dict)
		print(probabilities_10[0])
		return probabilities_10


