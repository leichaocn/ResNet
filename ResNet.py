# region 载入库,基础参数配置
import collections
import tensorflow as tf
from datetime import datetime
import math
import time
slim = tf.contrib.slim
num_batches=2
batch_size =2
'''
num_batches=100
batch_size =32
'''
# endregion

# region 数据预处理
height, width = 224, 224
inputs = tf.random_uniform((batch_size, height, width, 3))
# endregion

# region 计算图构造函数组

# region subsample():被bottleneck()调用
# 在残差学习单元bottleneck()中用到一次，
def subsample(inputs, factor, scope=None):
  if factor == 1:
    return inputs
  else:
    return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)
# endregion

# region conv2d_same():bottleneck()中的三层卷积的中间一层，resnet_v2()的最开始的卷积层
# 这个卷积函数，用在两个地方,。
# 一处是在残差单元的核心实现bottleneck()的中间一层，
# 另一处是在整个网络的核心实现resnet_v2()的最开始的卷积层用到过一次。
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
  if stride == 1:
    # 这一个卷积方式,被用了47次(层),分别是:
    # block1(总共3个单元)的头2个残差单元的(256,64,1)残差单元,每个单元的中间一层卷积.
    # block2(总共8个单元)的头7个残差单元的(512,128,1)残差单元,每个单元的中间一层卷积.
    # block3(总共36个单元)的头35个残差单元的(1024,256,1)残差单元,每个单元的中间一层卷积.
    # block4(总共3个单元)的全部3个残差单元的(2014,512,1)残差单元,每个单元的中间一层卷积.
    # 残差单元表示方法举例:(256,64,2)就表示一个残差单元内部的三层,
    # 256是最后一层的输出通道,64是前两层的输出通道,2是中间这一层的步长.
    # 即第一层(1*1,步长1,64),第二层(3*3,步长2,64),第三层(1*1,步长1,256)
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                       padding='SAME', scope=scope)
  else:
    # 这一个卷积方式,被用了4次(层),分别是:
    # 整个ResNet网络最开始的第一层卷积
    # block1(总共3个单元)的最后一个残差单元的(256,64,2)残差单元的中间一层卷积.
    # block2(总共8个单元)的最后一个残差单元的(512,128,2)残差单元的中间一层卷积.
    # block3(总共36个单元)的最后一个残差单元的(1024,256,2)残差单元的中间一层卷积.
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       padding='VALID', scope=scope)
# endregion

# region bottleneck():被stack_blocks_dense()调用(以block.unit_fn()的形式),实现单个残差单元
# 实现残差学习单元的核心函数
# 这个@不能去掉
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
    # 传入的outputs_collections=resnet_v2_152/_end_points

  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:

    # 用depth_in拿到inputs的最后一个维度的值，即输入的深度或通道数。
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

    #BN和预激活。
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

    # region 计算shortcut
    # 如果规定的输出深度等于输入深度，进行降采样
    if depth == depth_in:
      # 传进subsample()里的是args中的stride，如果stride是1，直接返回inputs，如果非1，则进行最大池化。
      shortcut = subsample(inputs, stride, 'shortcut')
    else:
        # 如果规定的输出深度与输入深度不同，我们用1*1的卷积改变其通道数，使得与输出通道一致。
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,#这里就是传进来的args中的depth和stride
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')
    #此时，shortcut的输出深度就等于depth了。
    # endregion

    # region 计算residual
    # 卷积 缺省激活函数为relu
    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,#这里就是传进来的args中的depth_bottleneck
                           scope='conv1')
    # 卷积 缺省激活函数为relu
    residual = conv2d_same(residual, depth_bottleneck, 3, stride,#这里就是传进来的args中的depth_bottleneck和stride
                                        scope='conv2')
    # 卷积 无激活函数
    # 因为bottleneck()会被调用50次,看似会有50个残差单元的结尾都没有激活函数.
    # 其实,这个relu就在下一个残差单元的预激活中,即使第50个残差单元输出后,在进入最后三个普通层时也会先预激活.
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,#这里就是传进来的args中的depth
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')
    # endregion

    # region 计算output
    #shortcut是输入直接降采样后的结果，或经过BN和预激活后经过1层卷积的结果
    #residual是输入经过BN和预激活，再经过三层卷积后的结果。
    output = shortcut + residual
    # endregion

    # 把output加入到集合outputs_collections中，给output一个别名sc.name，然后直接返回output
    # outputs_collections的值为resnet_v2_152/_end_points,而sc.name=resnet_v2_152
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)
# endregion,

# region stack_blocks_dense():被resnet_v2()调用,解析blocks元素,迭代调用bottleneck()生成残差学习模组
# 实现由50个残差单元组成的残差学习模组
# 对参数进行解析匹配，主要的残差单元结构是在内部调用bottleneck()时完成的。
# 这个@不能去掉。
@slim.add_arg_scope
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):
    # 其中outputs_collections=resnet_v2_152/_end_points
    # blocks是由四组类似Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)])组成的一个list。
  for block in blocks:
      # 第一次进循环的时候，block=Block(scope='block1',
      #                             unit_fn=<function bottleneck()>,
      #                             args=[(256, 64, 1)] * 2 + [(256, 64, 2)])
      # 第二次进循环,block就变成block2.以此类推.block3,block4.
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
        # block.scope的值即字符串block1,第二轮就是blocks2
      for i, unit in enumerate(block.args):
        # 第一次进循环，i=0，unit=(256, 64, 1)；
        # 第二次循环，i=1，unit=(256, 64, 1)；再后，i=2，unit=(256, 64,  2).
        # 循环3次，一个block（例如'block1'）就循环结束了。
        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          unit_depth, unit_depth_bottleneck, unit_stride = unit
          # 当i=0，unit=(256, 64, 1)，即残差单元输出层深度为256，瓶颈（中间层）深度为64，步长为1
          net = block.unit_fn(net,
                              depth=unit_depth,
                              depth_bottleneck=unit_depth_bottleneck,
                              stride=unit_stride)# block.unit_fn()就等于bottleneck()
      # 下面的slim.utils.collect_named_outputs()把net添加进集合outputs_collections中，然后以别名sc.name命名net，然后直接返回这个传入的net
      # outputs_collections=resnet_v2_152/_end_points，sc.name=resnet_v2_152
      # 到此一步，所有残差单元blocks1234就全部构造完成了，开始准备跳出本函数
      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)      
  return net
# endregion

# region resnet_v2():被resnet_v2_152()调用,实现整个构造结构,残差学习模组则是调用stack_blocks_dense()来实现
# 整体网络实现的核心函数
# 负责实现整个网络的的结构，在内部调用stack_blocks_dense()来实现残差学习单元
def resnet_v2(inputs,#传入inputs
              blocks,#传入blocks
              num_classes=None,#传入1000
              global_pool=True,#传入True
              include_root_block=True,#传入True
              reuse=None,#传入None
              scope=None):#scope=resnet_v2_152
 
  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
  #此处，sc.name就是scope，即resnet_v2_152，sc.original_name_scope就是resnet_v2_152/
    #定义一个end_points_collection，初始化为resnet_v2_152/_end_points
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         stack_blocks_dense],
                        outputs_collections=end_points_collection):
      #outputs_collections即被初始化为resnet_v2_152/_end_points，后面调用stack_blocks_dense()，bottleneck()时会用到。
      net = inputs

      # region 开始的2层
      # include_root_block为true,就是加上ResNet网络最开始部分的7*7卷积和最大池化.
      if include_root_block:
        with slim.arg_scope([slim.conv2d],
                            activation_fn=None, normalizer_fn=None):
          # 第一层,卷积.注意,这个卷积没有激活函数,激活函数相当于放在了残差学习模组中最开始的预激活中.
          net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
        # 第二层，最大池化
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
      # endregion

      # region 残差学习模组 (150层)
      net = stack_blocks_dense(net, blocks)
      # endregion

      # region 最后3层

      # 这个后正则,相当于给最后一个残差单元(即第50个)的输出补上了激活函数
      net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

      # 全局平均池化层
      if global_pool:
        # 全局平均池化层。效率比avg_pool()高
        net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

      # logits层，无激活函数，无BN
      if num_classes is not None:
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='logits')
      # end_points={'resnet_v2_152/_end_points':}
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      # 最后的softmax层
      if num_classes is not None:
        end_points['predictions'] = slim.softmax(net, scope='predictions')
      # endregion

      return net, end_points
# endregion

# region Block()类:用在resnet_v2_152()中,定义网络参数集合
# 这个类，虽然类体是空，但必须在冒号后有至少2个及以上单引号（或双引号）
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):''
# endregion

# region resnet_v2_152():传入输入数据,实例化用一个blocks列表,调用resnet_v2().
# ResNet网络的主函数
# 主要是传入残差学习单元的参数，整个网络结构的实现主要在resnet_v2()中实现。
def resnet_v2_152(inputs,#传入inputs
                  num_classes=None,#传入1000
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_152'):
  blocks = [
      Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      Block(
          'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
      Block(
          'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
      Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, global_pool,
                   include_root_block=True, reuse=reuse, scope=scope)
# endregion

# region resnet_arg_scope():参数配置函数,在执行函数中被作为实参传入with slim.arg_scope()中.
# 定义ResNet的默认参数。
# 在最终的测试例程中，被with slim.arg_scope()调用
def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
      'is_training': is_training,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc
# endregion

# endregion

# region 计算图执行函数
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s: 第 %d 步（batch）, 该步（batch）耗时 = %.3f 秒' %
                       (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s 经过 %d 步（batch）, 每步（batch）平均耗时为%.3f +/- %.3f 秒' %
           (datetime.now(), info_string, num_batches, mn, sd))
# endregion

# region 构造计算图
# with slim.arg_scope(),只有与@slim.add_arg_scope搭配，才能起到给指定函数传入缺省参数的目的。
# 其中，本代码中，@slim.add_arg_scope被加在stack_blocks_dense()和bottleneck()前面，指定配置这两个函数。
# resnet_arg_scope()在前面已经详细定义ResNet的默认参数，这里进行调用，作为实参传入slim.arg_scope()中。
with slim.arg_scope(resnet_arg_scope(is_training=False)):
   net, end_points = resnet_v2_152(inputs, 1000)
# 这里的1000即为输出类别总数
init = tf.global_variables_initializer()
# endregion

# region 执行计算图
sess = tf.Session()
sess.run(init)
time_tensorflow_run(sess, net, "Forward") 
# endregion
