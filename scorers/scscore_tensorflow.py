"""
This is the code for a standalone importable SCScorer model. It relies on tensorflow 
and simply reinitializes from a save file. 

One method dumps the trainable variables as numpy arrays, which then enables the 
standalone_model_numpy version of this class.
"""
# import tensorflow as tf
import math
import os
from functools import reduce
from typing import Callable

import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import tensorflow.compat.v1 as tf

from shared import project_dir

tf.get_logger().setLevel("ERROR")
tf.disable_v2_behavior()


def linear(input_, output_size, scope, reuse=False, init_bias=0.0):
    shape = input_.get_shape().as_list()
    stddev = min(1.0 / math.sqrt(shape[-1]), 0.1)
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            "Matrix",
            [shape[-1], output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev),
        )
    if init_bias is None:
        return tf.matmul(input_, W)
    with tf.variable_scope(scope, reuse=reuse):
        b = tf.get_variable(
            "bias", [output_size], initializer=tf.constant_initializer(init_bias)
        )
    return tf.matmul(input_, W) + b


def linearND(input_, output_size, scope, reuse=False, init_bias=0.0):
    shape = input_.get_shape().as_list()
    ndim = len(shape)
    stddev = min(1.0 / math.sqrt(shape[-1]), 0.1)
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            "Matrix",
            [shape[-1], output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev),
        )
    X_shape = tf.gather(tf.shape(input_), range(ndim - 1))
    target_shape = tf.concat([X_shape, [output_size]], 0)
    exp_input = tf.reshape(input_, [-1, shape[-1]])
    if init_bias is None:
        res = tf.matmul(exp_input, W)
    else:
        with tf.variable_scope(scope, reuse=reuse):
            b = tf.get_variable(
                "bias", [output_size], initializer=tf.constant_initializer(init_bias)
            )
        res = tf.matmul(exp_input, W) + b
    res = tf.reshape(res, target_shape)
    res.set_shape(shape[:-1] + [output_size])
    return res


score_scale = 5.0
min_separation = 0.25

FP_len = 1024
FP_rad = 2
batch_size = 2


class SCScorer:
    def __init__(self):
        self.session = tf.Session()

    def build(
        self,
        depth=5,
        hidden_size=300,
        score_scale=score_scale,
        FP_len=FP_len,
        FP_rad=FP_rad,
    ):
        self.FP_len = FP_len
        self.FP_rad = FP_rad
        self.input_mol = tf.placeholder(tf.float32, [batch_size * 2, FP_len])
        self.mol_hiddens = tf.nn.relu(
            linearND(self.input_mol, hidden_size, scope="encoder0")
        )
        for d in range(1, depth):
            self.mol_hiddens = tf.nn.relu(
                linearND(self.mol_hiddens, hidden_size, scope="encoder%i" % d)
            )

        self.score_sum = linearND(self.mol_hiddens, 1, scope="score_sum")
        self.score_sum = tf.squeeze(self.score_sum)
        self.score = 1.0 + (score_scale - 1.0) * tf.nn.sigmoid(self.score_sum)

        tf.global_variables_initializer().run(session=self.session)

        def size_func(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())

        n = sum(size_func(v) for v in tf.trainable_variables())
        # print "Model size: %dK" % (n/1000,)

        self.coord = tf.train.Coordinator()
        return self

    def restore(self, model_path, checkpoint="final"):
        self.saver = tf.train.Saver(max_to_keep=None)
        restore_path = os.path.join(model_path, "model.%s" % checkpoint)
        self.saver.restore(self.session, restore_path)
        # print("Restored values from latest saved file ({})".format(restore_path))

        if "uint8" in model_path or "counts" in model_path:

            def mol_to_fp(self, mol):
                if mol is None:
                    return np.array((self.FP_len,), dtype=np.uint8)
                fp = AllChem.GetMorganFingerprint(
                    mol, self.FP_rad, useChirality=True
                )  # uitnsparsevect
                fp_folded = np.zeros((self.FP_len,), dtype=np.uint8)
                for k, v in fp.GetNonzeroElements().iteritems():
                    fp_folded[k % self.FP_len] += v
                return np.array(fp_folded)

        else:

            def mol_to_fp(self, mol):
                if mol is None:
                    return np.zeros((self.FP_len,), dtype=np.float32)
                return np.array(
                    AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.FP_rad, nBits=self.FP_len, useChirality=True
                    ),
                    dtype=np.bool,
                )

        self.mol_to_fp = mol_to_fp
        return self

    def smi_to_fp(self, smi):
        if not smi:
            return np.zeros((self.FP_len,), dtype=np.float32)
        return self.mol_to_fp(self, Chem.MolFromSmiles(smi))

    def get_score_from_smi(self, smi="", v=False):
        if not smi:
            return ("", 0.0)
        src_batch = [smi]
        while len(src_batch) != (batch_size * 2):  # round out last batch
            src_batch.append("")
        src_mols = np.array(list(map(self.smi_to_fp, src_batch)), dtype=np.float32)
        if sum(sum(src_mols)) == 0:
            if v:
                print("Could not get fingerprint?")
            cur_score = [0.0]
        else:
            # Run
            (cur_score,) = self.session.run(
                [self.score], feed_dict={self.input_mol: src_mols,}
            )
            if v:
                print("Score: {}".format(cur_score[0]))
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        else:
            smi = ""
        return (smi, cur_score[0])

    def dump_to_numpy_arrays(self, dump_path):
        import cPickle as pickle

        with open(dump_path, "wb") as fid:
            pickle.dump(
                [v.eval(session=self.session) for v in tf.trainable_variables()],
                fid,
                -1,
            )


def get_sc_scorer() -> Callable[[str], float]:
    model = SCScorer()
    model.build()
    model.restore(
        os.path.join(project_dir, "data", "sc_models", "full_reaxys_model_1024bool"),
        "ckpt-10654",
    )

    def scorer(smiles: str) -> float:
        _, score = model.get_score_from_smi(smiles)
        return score.item()

    return scorer
