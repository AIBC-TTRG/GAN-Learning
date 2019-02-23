from trainer import *
from models256 import *
from datasets import deepfashion
# import gabor 
from gabor import *

class PG2_256(PG2):
    def __init__(self, config):
        print('PG2_256_common_init')
        self._common_init(config)
        self.keypoint_num = 18
        self.D_arch = config.D_arch     #DCGAN

        if ('deepfashion' in config.dataset.lower()) or ('df' in config.dataset.lower()):
            if config.is_train:
                self.dataset_obj = deepfashion.get_split('train', config.data_path, data_name='DeepFashion')
            else:
                self.dataset_obj = deepfashion.get_split('test', config.data_path, data_name='DeepFashion')

        self.x, self.x_target, self.pose, self.pose_target, self.mask, self.mask_target = self._load_batch_pair_pose(self.dataset_obj)

    def _getDiscriminator(self, wgan_gp, arch='DCGAN'):
        print('PG2_256__getDiscriminator')
        """
        Choose which generator and discriminator architecture to use by
        uncommenting one of these lines.
        """        
        if 'DCGAN'==arch:
            # Baseline (G: DCGAN, D: DCGAN)
            return wgan_gp.DCGANDiscriminator_256
        raise Exception('You must choose an architecture!')

    def _gan_loss(self, wgan_gp, Discriminator, disc_real, disc_fake, arch='DCGAN'):
        print('PG2_256___gan_loss')
        if wgan_gp.MODE == 'dcgan':
            if 'DCGAN'==arch:
                gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                  labels=tf.ones_like(disc_fake)))
                disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                    labels=tf.zeros_like(disc_fake)))
                disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                    labels=tf.ones_like(disc_real)))                    
                disc_cost /= 2.
        else:
            raise Exception()
        return gen_cost, disc_cost

    def build_model(self):
        print('PG2_256__build_model')
        G1, DiffMap, self.G_var1, self.G_var2  = GeneratorCNN_Pose_UAEAfterResidual_UAEnoFCAfterNoise_256(
                self.x, self.pose_target, 
                self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, noise_dim=0, reuse=False)

        # G11,G3, self.G_var11, self.G_var3  = GeneratorCNN_Pose_UAEAfterResidual_UAEnoFCAfterNoise_1(
        #         self.x, self.pose_target, 
        #         self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, noise_dim=0, reuse=False)


        
        G2 = G1 + DiffMap
        print('====================')
        print(G1)
        print('====================')
        print(DiffMap)
        print('====================')
        print(G2)

        print('====================')
        print(self.x_target)

        # G3 = Gabor_u4v6(self.x_target)
        # print('G3=====')
        # print(G3)
        # G2 = G1 + DiffMap + G3
        self.G1 = denorm_img(G1, self.data_format)
        self.G2 = denorm_img(G2, self.data_format)
        # self.G3 = denorm_img(G3, self.data_format)
        self.G = self.G2
        self.DiffMap = denorm_img(DiffMap, self.data_format)
        self.wgan_gp = WGAN_GP(DATA_DIR='', MODE='dcgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, LAMBDA=10, G_OUTPUT_DIM=256*256*3)
        Dis = self._getDiscriminator(self.wgan_gp, arch=self.D_arch)

        triplet = tf.concat([self.x_target, G2, self.x], 0)

        ## WGAN-GP code uses NCHW
        self.D_z = Dis(tf.transpose( triplet, [0,3,1,2] ), input_dim=3)
        self.D_var = lib.params_with_name('Discriminator.')

        D_z_pos_x_target, D_z_neg_g2, D_z_neg_x = tf.split(self.D_z, 3)
        D_z_pos = D_z_pos_x_target
        D_z_neg = tf.concat([D_z_neg_g2, D_z_neg_x], 0)


        self.PoseMaskLoss1 = tf.reduce_mean(tf.abs(G1 - self.x_target) * (self.mask_target))
        self.g_loss1 = tf.reduce_mean(tf.abs(G1-self.x_target)) + self.PoseMaskLoss1

        self.g_loss2, self.d_loss = self._gan_loss(self.wgan_gp, Dis, D_z_pos, D_z_neg, arch=self.D_arch)
        self.PoseMaskLoss2 = tf.reduce_mean(tf.abs(G2 - self.x_target) * (self.mask_target))
        self.L1Loss2 = tf.reduce_mean(tf.abs(G2 - self.x_target)) + self.PoseMaskLoss2
        self.g_loss2 += self.L1Loss2 * 50

        self.g_optim1, self.g_optim2, self.d_optim, self.clip_disc_weights = self._getOptimizer(self.wgan_gp, 
                                self.g_loss1, self.g_loss2, self.d_loss, self.G_var1,self.G_var2, self.D_var)
        self.summary_op = tf.summary.merge([
            tf.summary.image("G1", self.G1),
            tf.summary.image("G2", self.G2),
            # tf.summary.image("G3", self.G3),
            tf.summary.image("DiffMap", self.DiffMap),
            tf.summary.scalar("loss/PoseMaskLoss1", self.PoseMaskLoss1),
            tf.summary.scalar("loss/PoseMaskLoss2", self.PoseMaskLoss2),
            tf.summary.scalar("loss/L1Loss2", self.L1Loss2),
            tf.summary.scalar("loss/g_loss1", self.g_loss1),
            tf.summary.scalar("loss/g_loss2", self.g_loss2),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])

    def _load_batch_pair_pose(self, dataset, mode='coordSolid'):
        print('PG2_256__load_batch_pair_pose')
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=8)

        image_raw_0, image_raw_1, label, pose_0, pose_1, mask_0, mask_1 = data_provider.get([
            'image_raw_0', 'image_raw_1', 'label', 'pose_sparse_r4_0', 'pose_sparse_r4_1', 'pose_mask_r4_0', 'pose_mask_r4_1'])
        
        print('***************************')
        print image_raw_0

        G3 = Gabor_u4v6(image_raw_0)
        print('G3=====')
        print(G3)

        pose_0 = sparse_ops.sparse_tensor_to_dense(pose_0, default_value=0, validate_indices=False)
        pose_1 = sparse_ops.sparse_tensor_to_dense(pose_1, default_value=0, validate_indices=False)

        image_raw_0 = tf.reshape(image_raw_0, [256, 256, 3])        
        image_raw_1 = tf.reshape(image_raw_1, [256, 256, 3]) 
        pose_0 = tf.cast(tf.reshape(pose_0, [256, 256, self.keypoint_num]), tf.float32)
        pose_1 = tf.cast(tf.reshape(pose_1, [256, 256, self.keypoint_num]), tf.float32)
        mask_0 = tf.cast(tf.reshape(mask_0, [256, 256, 1]), tf.float32)
        mask_1 = tf.cast(tf.reshape(mask_1, [256, 256, 1]), tf.float32)

        images_0, images_1, poses_0, poses_1, masks_0, masks_1 = tf.train.batch([image_raw_0, image_raw_1, pose_0, pose_1, mask_0, mask_1], 
                    batch_size=self.batch_size, num_threads=self.num_threads, capacity=self.capacityCoff * self.batch_size)

        images_0 = utils_wgan.process_image(tf.to_float(images_0), 127.5, 127.5)
        images_1 = utils_wgan.process_image(tf.to_float(images_1), 127.5, 127.5)
        poses_0 = poses_0*2-1
        poses_1 = poses_1*2-1
        return images_0, images_1, poses_0, poses_1, masks_0, masks_1

def GeneratorCNN_Pose_UAEAfterResidual_1(x, pose_target, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, min_fea_map_H=8, noise_dim=0, reuse=False):
    with tf.variable_scope("G") as vs:
        if pose_target is not None:
            if data_format == 'NCHW':
                x = tf.concat([x, pose_target], 1)
            elif data_format == 'NHWC':
                x = tf.concat([x, pose_target], 3)

        # pdb.set_trace()
        # Encoder
        encoder_layer_list = []
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)

        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            # channel_num = x.get_shape()[-1]
            res = x
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = x + res
            encoder_layer_list.append(x)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, hidden_num * (idx + 2), 3, 2, activation_fn=activation_fn, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        x = tf.reshape(x, [-1, np.prod([min_fea_map_H, min_fea_map_H, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)
        if noise_dim>0:
            noise = tf.random_uniform(
                (tf.shape(z)[0], noise_dim), minval=-1.0, maxval=1.0)
            z = tf.concat([z, noise], 1)

        # Decoder
        x = slim.fully_connected(z, np.prod([min_fea_map_H, min_fea_map_H, hidden_num]), activation_fn=None)
        x = reshape(x, min_fea_map_H, min_fea_map_H, hidden_num, data_format)
        
        for idx in range(repeat_num):
            # pdb.set_trace()
            x = tf.concat([x, encoder_layer_list[repeat_num-1-idx]], axis=-1)
            res = x
            # channel_num = hidden_num * (repeat_num-idx)
            channel_num = x.get_shape()[-1]
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = x + res
            if idx < repeat_num - 1:
                # x = slim.layers.conv2d_transpose(x, hidden_num * (repeat_num-idx-1), 3, 2, activation_fn=activation_fn, data_format=data_format)
                x = upscale(x, 2, data_format)
                x = slim.conv2d(x, hidden_num * (repeat_num-idx-1), 1, 1, activation_fn=activation_fn, data_format=data_format)


        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables


def GeneratorCNN_Pose_UAEAfterResidual_UAEnoFCAfterNoise_1(x, pose_target, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, noise_dim=64, reuse=False):
    with tf.variable_scope("Pose_AE") as vs1:
        out1, _, var1 = GeneratorCNN_Pose_UAEAfterResidual_1(x, pose_target, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=activation_fn, noise_dim=0, reuse=False)
    with tf.variable_scope("UAEnoFC") as vs2:
        out2, var2 = UAE_noFC_AfterNoise1(x, input_channel, z_num, repeat_num-2, hidden_num, data_format, noise_dim=noise_dim, activation_fn=activation_fn, reuse=False)
    # with tf.variable_scope("UAEnoFC") as vs2:
    #     out3, var3 = UAE_noFC_AfterNoise1(x, input_channel, z_num, repeat_num-2, hidden_num, data_format, noise_dim=noise_dim, activation_fn=activation_fn, reuse=False)
    return out1,out2,var1,var2