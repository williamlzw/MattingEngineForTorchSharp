using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace MattingNetHelp
{
    public class ConvBNRelu : Module<Tensor, Tensor>
    {
        private readonly Conv2d conv;
        private readonly BatchNorm2d bn;
        private readonly ReLU relu;

        public ConvBNRelu(long in_channels, long out_channels, (long, long) kernel_size, (long, long) stride) : base("ConvBNRelu")
        {
            var (k1, k2) = kernel_size;
            var pad = k1 / 2;
            conv = Conv2d(in_channels, out_channels, kernel_size, stride: stride, dilation: (1, 1), padding: (pad, pad), bias: false);
            bn = BatchNorm2d(out_channels);
            relu = ReLU();
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = relu.forward(bn.forward(conv.forward(input)));
            return x;
        }
    }

    public class ConvBNReLU : Module<Tensor, Tensor>
    {
        private readonly Conv2d _conv;
        private readonly BatchNorm2d _batch_norm;
        private readonly ReLU _relu;

        public ConvBNReLU(long in_channels, long out_channels, (long, long) kernel_size, (long, long) stride, long padding = 0, long groups = 1, bool bias = false) : base("ConvBNReLU")
        {
            _conv = Conv2d(in_channels, out_channels, kernel_size, stride: stride, dilation: (1, 1), padding: (padding, padding), groups: groups, bias: bias);
            _batch_norm = BatchNorm2d(out_channels);
            _relu = ReLU();
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = _relu.forward(_batch_norm.forward(_conv.forward(input)));
            return x;
        }
    }

    public class ConvBN : Module<Tensor, Tensor>
    {
        private readonly Conv2d _conv;
        private readonly BatchNorm2d _batch_norm;

        public ConvBN(long in_channels, long out_channels, (long, long) kernel_size, (long, long) stride, long padding = 0, bool bias = false) : base("ConvBN")
        {
            _conv = Conv2d(in_channels, out_channels, kernel_size, stride: stride, dilation: (1, 1), padding: (padding, padding), bias: bias);
            _batch_norm = BatchNorm2d(out_channels);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = _batch_norm.forward(_conv.forward(input));
            return x;
        }
    }

    public class Conv2DBN : Module<Tensor, Tensor>
    {
        private readonly Conv2d c;
        private readonly BatchNorm2d bn;

        public Conv2DBN(long in_channels, long out_channels, (long, long) kernel_size, (long, long) stride) : base("Conv2DBN")
        {
            c = Conv2d(in_channels, out_channels, kernel_size, stride: stride, dilation: (1, 1), bias: false);
            bn = BatchNorm2d(out_channels);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = bn.forward(c.forward(input));
            return x;
        }
    }

    public class PyramidPoolAgg : Module<List<Tensor>, Tensor>
    {
        private readonly long _stride;
        private readonly BatchNorm2d _batch_norm;

        public PyramidPoolAgg(long stride) : base("PyramidPoolAgg")
        {
            _stride = stride;
            RegisterComponents();
        }

        public override Tensor forward(List<Tensor> input)
        {
            var ks = (long)Math.Pow(2, input.Count);
            var stride = (long)Math.Pow(_stride, input.Count);
            List<Tensor> outList = new List<Tensor>();
            foreach (var x in input)
            {
                var x1 = torch.nn.functional.avg_pool2d(x, ks, stride);
                ks = ks / 2;
                stride = stride / 2;
                outList.Add(x1);
            }
            var ret = torch.cat(outList, 1);
            return ret;
        }
    }

    public class Attention : Module<Tensor, Tensor>
    {
        private readonly Conv2DBN to_q;
        private readonly Conv2DBN to_k;
        private readonly Conv2DBN to_v;
        private readonly Sequential proj;
        private readonly long _num_heads;
        private readonly double _scale;
        private readonly long _key_dim;
        private readonly long _nh_kd;
        private readonly long _dh;
        private readonly long _d;

        public Attention(long dim, long key_dim, long num_heads, long attn_ratio, string actType) : base("Attention")
        {
            _num_heads = num_heads;
            _scale = Math.Pow(key_dim, -0.5);
            _key_dim = key_dim;
            _nh_kd = key_dim * num_heads;
            _d = attn_ratio * key_dim;
            _dh = attn_ratio * key_dim * num_heads;
            to_q = new Conv2DBN(dim, _nh_kd, (1, 1), (1, 1));
            to_k = new Conv2DBN(dim, _nh_kd, (1, 1), (1, 1));
            to_v = new Conv2DBN(dim, _dh, (1, 1), (1, 1));
            proj = Sequential();
            if (actType == "relu")
            {
                proj.append(ReLU());
            }
            else if (actType == "relu6")
            {
                proj.append(ReLU6());
            }
            proj.append(new Conv2DBN(_dh, dim, (1, 1), (1, 1)));
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var height = input.shape[2];
            var width = input.shape[3];
            var qq = to_q.forward(input).reshape(1, _num_heads, _key_dim, -1).permute(0, 1, 3, 2);
            var kk = to_k.forward(input).reshape(1, _num_heads, _key_dim, -1);
            var vv = to_v.forward(input).reshape(1, _num_heads, _d, -1).permute(0, 1, 3, 2);
            var attn = torch.matmul(qq, kk);
            attn = torch.nn.functional.softmax(attn, -1);
            var xx = torch.matmul(attn, vv);
            xx = xx.permute(0, 1, 3, 2).reshape(1, _dh, height, width);
            xx = proj.forward(xx);
            return xx;
        }
    }

    public class DropPath : Module<Tensor, Tensor>
    {
        private readonly double _drop_prob;

        public DropPath(double drop_prob) : base("DropPath")
        {
            _drop_prob = drop_prob;
            RegisterComponents();
        }
        public override Tensor forward(Tensor input)
        {
            return input;
        }
    }

    public class MLP : Module<Tensor, Tensor>
    {
        private readonly double _drop_prob;
        private readonly Conv2DBN fc1;
        private readonly Conv2DBN fc2;
        private readonly Conv2d dwconv;
        private readonly Module<Tensor, Tensor> act;
        private readonly Dropout drop;

        public MLP(long in_features, long hidden_features, long out_features, string actType = "relu", double dropRatio = 0) : base("MLP")
        {
            fc1 = new Conv2DBN(in_features, hidden_features, (1, 1), (1, 1));
            dwconv = Conv2d(hidden_features, hidden_features, 3, 1, 1, groups: hidden_features);
            if (actType == "relu")
            {
                act = ReLU();
            }
            else if (actType == "relu6")
            {
                act = ReLU6();
            }
            fc2 = new Conv2DBN(hidden_features, out_features, (1, 1), (1, 1));
            drop = Dropout(dropRatio);
            RegisterComponents();
        }
        public override Tensor forward(Tensor input)
        {
            var x = fc1.forward(input);
            x = dwconv.forward(x);
            x = act.forward(x);
            x = drop.forward(x);
            x = fc2.forward(x);
            x = drop.forward(x);
            return x;
        }
    }

    public class Block : Module<Tensor, Tensor>
    {
        private readonly Attention attn;
        private readonly DropPath drop_path;
        private readonly MLP mlp;
        private readonly long _dim;
        private readonly long _num_heads;
        private readonly long _mlp_ratio;

        public Block(long dim, long key_dim, long num_heads, long mlp_ratio = 4, long attn_ratio = 2, double dropRatio = 0, double dropPath = 0, string actType = "relu") : base("Block")
        {
            _dim = dim;
            _num_heads = num_heads;
            _mlp_ratio = mlp_ratio;
            attn = new Attention(dim, key_dim, num_heads, attn_ratio, actType);
            drop_path = new DropPath(dropPath);
            var mlp_hidden_dim = dim * mlp_ratio;
            mlp = new MLP(dim, mlp_hidden_dim, dim, actType, dropRatio);
            RegisterComponents();
        }
        public override Tensor forward(Tensor input)
        {
            var h = input;
            var x = attn.forward(input);
            x = drop_path.forward(x);
            x = h + x;
            h = x;
            x = mlp.forward(x);
            x = drop_path.forward(x);
            x = x + h;
            return x;
        }
    }

    public class BasicLayer : Module<Tensor, Tensor>
    {
        private readonly long _block_num;
        private readonly ModuleList<Module<Tensor, Tensor>> transformer_blocks;

        public BasicLayer(long block_num, long embedding_dim, long key_dim, long num_heads, long mlp_ratio = 4, long attn_ratio = 2, double dropRatio = 0, double dropPath = 0, string actType = "relu") : base("BasicLayer")
        {
            _block_num = block_num;
            transformer_blocks = new ModuleList<Module<Tensor, Tensor>>();
            for (int i = 0; i < block_num; i++)
            {
                transformer_blocks.Add(new Block(embedding_dim, key_dim, num_heads, mlp_ratio, attn_ratio, dropRatio, dropPath, actType));
            }
            RegisterComponents();
        }
        public override Tensor forward(Tensor input)
        {
            var x = input;
            foreach (var index in transformer_blocks)
            {
                x = index.forward(x);
            }
            return x;
        }
    }

    public class MLFF : Module<List<Tensor>, long[], Tensor>
    {
        private readonly ModuleList<Module<Tensor, Tensor>> pwconvs;
        private readonly ModuleList<Module<Tensor, Tensor>> dwconvs;
        private readonly Sequential conv_atten;
        private readonly ConvBNReLU conv_out;

        public MLFF(List<long> in_channels, List<long> mid_channels, long out_channel) : base("MLFF")
        {
            pwconvs = new ModuleList<Module<Tensor, Tensor>>();
            dwconvs = new ModuleList<Module<Tensor, Tensor>>();

            for (int i = 0; i < in_channels.Count; i++)
            {
                var in_channel = in_channels[i];
                var mid_channel = mid_channels[i];
                pwconvs.Add(new ConvBN(in_channel, mid_channel, (1, 1), (1, 1)));
                dwconvs.Add(new ConvBNReLU(mid_channel, mid_channel, (3, 3), (1, 1), padding: 1, groups: mid_channel));
            }
            var num_feas = in_channels.Count;
            conv_atten = Sequential(
                new ConvBNReLU(2 * num_feas, num_feas, (3, 3), (1, 1), padding: 1),
                new ConvBN(num_feas, num_feas, (3, 3), (1, 1), 1));
            var in_chan = mid_channels.Sum();
            conv_out = new ConvBNReLU(in_chan, out_channel, (3, 3), (1, 1), 1);
            RegisterComponents();
        }


        private Tensor AvgMaxReduceChannelHelperConCat(Tensor x)
        {
            var mean_value = torch.mean(x, new long[] { 1 }, keepdim: true);
            var (max_value, max_index) = torch.max(x, 1, keepdim: true);
            List<Tensor> cat = new List<Tensor>();
            cat.Add(mean_value);
            cat.Add(max_value);
            return torch.cat(cat, 1);
        }

        private List<Tensor> AvgMaxReduceChannelHelper(Tensor x)
        {
            var mean_value = torch.mean(x, new long[] { 1 }, keepdim: true);
            var (max_value, max_index) = torch.max(x, 1, keepdim: true);
            List<Tensor> cat = new List<Tensor>();
            cat.Add(mean_value);
            cat.Add(max_value);
            return cat;
        }

        private Tensor AvgMaxReduceChannel(List<Tensor> x)
        {
            if (x.Count == 1)
            {
                return AvgMaxReduceChannelHelperConCat(x[0]);
            }
            else
            {
                List<Tensor> res = new List<Tensor>();
                foreach (var xi in x)
                {
                    res.AddRange(AvgMaxReduceChannelHelper(xi));
                }
                return torch.cat(res, 1);
            }
        }

        public override Tensor forward(List<Tensor> inputs, long[] shape)
        {
            List<Tensor> feas = new List<Tensor>();
            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                var x = pwconvs[i].forward(input);
                x = torch.nn.functional.interpolate(x, shape, mode: InterpolationMode.Bilinear, align_corners: false);
                x = dwconvs[i].forward(x);
                feas.Add(x);
            }

            var atten = AvgMaxReduceChannel(feas);
            atten = torch.nn.functional.sigmoid(conv_atten.forward(atten));
            List<Tensor> feas_att = new List<Tensor>();
            for (int i = 0; i < feas.Count; i++)
            {
                var fea = feas[i];
                fea = fea * (atten[TensorIndex.Colon, TensorIndex.Single(i), TensorIndex.Colon, TensorIndex.Colon].unsqueeze(1));
                feas_att.Add(fea);
            }
            var sum = torch.cat(feas_att, 1);
            var ret = conv_out.forward(sum);
            return ret;
        }
    }

    public class MattingHead : Module<Tensor, Tensor>
    {
        private readonly ConvBNReLU conv;
        private readonly ModuleList<Module<Tensor, Tensor>> mid_conv;
        private readonly Conv2d conv_out;

        public MattingHead(long in_chan, long mid_chan, long mid_num = 1, long out_channels = 1) : base("MattingHead")
        {
            conv = new ConvBNReLU(in_chan, mid_chan, (3, 3), (1, 1), 1);
            mid_conv = new ModuleList<Module<Tensor, Tensor>>();
            for (int i = 0; i < mid_num - 1; i++)
            {
                mid_conv.Add(new ConvBNReLU(mid_chan, mid_chan, (3, 3), (1, 1), 1));
            }
            conv_out = Conv2d(mid_chan, out_channels, kernelSize: 1, bias: false);
            RegisterComponents();
        }
        public override Tensor forward(Tensor input)
        {
            var x = conv.forward(input);
            foreach (var mid in mid_conv)
            {
                x = mid.forward(x);
            }
            x = conv_out.forward(x);
            x = torch.nn.functional.sigmoid(x);
            return x;
        }
    }

    public class DoublePyramidPoolModule : Module<List<Tensor>, Tensor>
    {
        private readonly long _mid_channel;
        private readonly long _mlp_ratio;
        private readonly long _attn_ratio;
        private readonly List<long> _len_trans;
        private readonly PyramidPoolAgg pp1;
        private readonly ConvBN conv_mid;
        private readonly ModuleList<Module<Tensor, Tensor>> pp2;
        private readonly ConvBNReLU conv_out;

        public DoublePyramidPoolModule(long stride, long input_channel, long mid_channel, long output_channel, List<long> bin_sizes, long len_trans = 1, long mlp_ratio = 4, long attn_ratio = 2, double dropRatio = 0, double dropPath = 0) : base("DoublePyramidPoolModule")
        {
            _mid_channel = mid_channel;
            _mlp_ratio = mlp_ratio;
            _attn_ratio = attn_ratio;
            _len_trans = new List<long>(new long[] { 1, 1, 1 });
            pp1 = new PyramidPoolAgg(stride);
            conv_mid = new ConvBN(input_channel, mid_channel, (1, 1), (1, 1), bias: true);
            pp2 = new ModuleList<Module<Tensor, Tensor>>();
            for (int i = 0; i < 3; i++)
            {
                var size = bin_sizes[i];
                var block_num = _len_trans[i];
                pp2.Add(MakeStage(mid_channel, size, block_num));
            }
            var in_chan = mid_channel;
            conv_out = new ConvBNReLU(in_chan, output_channel, (1, 1), (1, 1), bias: true);
            RegisterComponents();
        }

        private Sequential MakeStage(long embdeding_channels, long size, long block_num)
        {
            var prior = AdaptiveAvgPool2d(size);
            Module<Tensor, Tensor> trans = null;
            if (size == 1)
            {
                trans = new ConvBNReLU(embdeding_channels, embdeding_channels, (1, 1), (1, 1));
            }
            else
            {
                trans = new BasicLayer(block_num, embdeding_channels, 16, 8, _mlp_ratio, _attn_ratio, 0, 0, "relu6");
            }
            return Sequential(prior, trans);
        }

        public override Tensor forward(List<Tensor> input)
        {
            var x = pp1.forward(input);
            var pp2_input = conv_mid.forward(x);

            List<Tensor> cat_layers = new List<Tensor>();
            foreach (var stage in pp2)
            {
                x = stage.forward(pp2_input);
                long[] shape = { pp2_input.shape[2], pp2_input.shape[3] };
                x = torch.nn.functional.interpolate(x, shape, mode: InterpolationMode.Bilinear, align_corners: false);
                cat_layers.Add(x);
            }
            cat_layers.Add(pp2_input);
            cat_layers.Reverse();
            torch.Tensor cat = 0;
            foreach (var index in cat_layers)
            {
                cat = cat + index;
            }
            var ret = conv_out.forward(cat);

            return ret;
        }
    }

    public class CatBottleneck : Module<Tensor, Tensor>
    {
        private readonly ModuleList<Module<Tensor, Tensor>> conv_list;
        private readonly Sequential avd_layer;
        private readonly AvgPool2d skip;
        private readonly long _stride;
        public CatBottleneck(long in_channels, long out_channels, long block_num = 3, long stride = 1) : base("CatBottleneck")
        {
            conv_list = new ModuleList<Module<Tensor, Tensor>>();
            _stride = stride;
            if (stride == 2)
            {
                avd_layer = Sequential(
                    Conv2d(out_channels / 2, out_channels / 2, kernelSize: 3, stride: 2, padding: 1, groups: out_channels / 2, bias: false),
                    BatchNorm2d(out_channels / 2)
                    );
                skip = AvgPool2d(kernel_size: 3, stride: 2);
                stride = 1;
            }
            for (int idx = 0; idx < block_num; idx++)
            {
                if (idx == 0)
                {
                    conv_list.Add(new ConvBNRelu(in_channels, out_channels / 2, (1, 1), (1, 1)));
                }
                else if (idx == 1 && block_num == 2)
                {
                    conv_list.Add(new ConvBNRelu(out_channels / 2, out_channels / 2, (3, 3), (stride, stride)));
                }
                else if (idx == 1 && block_num > 2)
                {
                    conv_list.Add(new ConvBNRelu(out_channels / 2, out_channels / 4, (3, 3), (stride, stride)));
                }
                else if (idx < block_num - 1)
                {
                    conv_list.Add(new ConvBNRelu(out_channels / (int)Math.Pow(2, idx), out_channels / (int)Math.Pow(2, idx + 1), (3, 3), (1, 1)));
                }
                else
                {
                    conv_list.Add(new ConvBNRelu(out_channels / (int)Math.Pow(2, idx), out_channels / (int)Math.Pow(2, idx), (3, 3), (1, 1)));
                }
            }
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var out1 = conv_list[0].forward(input);
            torch.Tensor x = null;
            List<Tensor> outList = new List<Tensor>();
            for (int idx = 1; idx < conv_list.Count; idx++)
            {
                if (idx == 1)
                {
                    if (_stride == 2)
                    {
                        x = conv_list[idx].forward(avd_layer.forward(out1));
                    }
                    else
                    {
                        x = conv_list[idx].forward(out1);
                    }
                }
                else
                {
                    x = conv_list[idx].forward(x);
                }
                outList.Add(x);
            }

            if (_stride == 2)
            {
                out1 = torch.nn.functional.avg_pool2d(out1, 3, 2, 1);
            }

            outList.Insert(0, out1);
            var ret = torch.cat(outList, dim: 1);
            return ret;
        }
    }

    public class AddBottleneck : Module<Tensor, Tensor>
    {
        private readonly ModuleList<Module<Tensor, Tensor>> conv_list;
        private readonly Sequential avd_layer;
        private readonly Sequential skip;
        private readonly long _stride;
        public AddBottleneck(long in_channels, long out_channels, long block_num = 3, long stride = 1) : base("AddBottleneck")
        {
            conv_list = new ModuleList<Module<Tensor, Tensor>>();
            _stride = stride;
            if (stride == 2)
            {
                avd_layer = Sequential(
                    Conv2d(out_channels / 2, out_channels / 2, kernelSize: 3, stride: 2, padding: 1, groups: out_channels / 2, bias: false),
                    BatchNorm2d(out_channels / 2)
                    );
                skip = Sequential(
                    Conv2d(in_channels, in_channels, kernelSize: 3, stride: 2, padding: 1, groups: in_channels, bias: false),
                    BatchNorm2d(in_channels),
                    Conv2d(in_channels, out_channels, kernelSize: 1, bias: false),
                    BatchNorm2d(out_channels)
                    );
                stride = 1;
            }
            for (int idx = 0; idx < block_num; idx++)
            {
                if (idx == 0)
                {
                    conv_list.Add(new ConvBNRelu(in_channels, out_channels / 2, (1, 1), (1, 1)));
                }
                else if (idx == 1 && block_num == 2)
                {
                    conv_list.Add(new ConvBNRelu(out_channels / 2, out_channels / 2, (3, 3), (stride, stride)));
                }
                else if (idx == 1 && block_num > 2)
                {
                    conv_list.Add(new ConvBNRelu(out_channels / 2, out_channels / 4, (3, 3), (stride, stride)));
                }
                else if (idx < block_num - 1)
                {
                    conv_list.Add(new ConvBNRelu(out_channels / (int)Math.Pow(2, idx), out_channels / (int)Math.Pow(2, idx + 1), (3, 3), (1, 1)));
                }
                else
                {
                    conv_list.Add(new ConvBNRelu(out_channels / (int)Math.Pow(2, idx), out_channels / (int)Math.Pow(2, idx), (3, 3), (1, 1)));
                }
            }
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            torch.Tensor x = input;
            List<Tensor> outList = new List<Tensor>();
            for (int idx = 0; idx < conv_list.Count; idx++)
            {
                if (idx == 0 && _stride == 2)
                {
                    x = avd_layer.forward(conv_list[idx].forward(x));
                }
                else
                {
                    x = conv_list[idx].forward(x);
                }
                outList.Add(x);
            }
            if (_stride == 2)
            {
                x = skip.forward(x);
            }
            var ret = torch.cat(outList, dim: 1) + x;
            return ret;
        }
    }

    public class STDCNet : Module<Tensor, List<Tensor>>
    {
        private readonly ModuleList<Module<Tensor, Tensor>> features;
        private readonly List<long> _layers;
        public readonly List<long> _feat_channels;

        public STDCNet(List<long> layers, long base_channels = 64, long block_num = 4, long in_channels = 3) : base("STDCNet")
        {
            _layers = layers;
            _feat_channels = new List<long>();
            _feat_channels.Add(base_channels / 2);
            _feat_channels.Add(base_channels);
            _feat_channels.Add(base_channels * 4);
            _feat_channels.Add(base_channels * 8);
            _feat_channels.Add(base_channels * 16);
            features = MakeLayers(in_channels, base_channels, layers, block_num);
            RegisterComponents();
        }

        private ModuleList<Module<Tensor, Tensor>> MakeLayers(long in_channels, long base_channels, List<long> layers, long block_num)
        {
            var features = new ModuleList<Module<Tensor, Tensor>>();
            features.Add(new ConvBNRelu(in_channels, base_channels / 2, (3, 3), (2, 2)));
            features.Add(new ConvBNRelu(base_channels / 2, base_channels, (3, 3), (2, 2)));
            for (int i = 0; i < layers.Count; i++)
            {
                var layer = layers[i];
                for (int j = 0; j < layer; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        features.Add(new CatBottleneck(base_channels, base_channels * 4, block_num: block_num, stride: 2));
                    }
                    else if (j == 0)
                    {
                        features.Add(new CatBottleneck(base_channels * (long)Math.Pow(2, i + 1), base_channels * (long)Math.Pow(2, i + 2), block_num: block_num, stride: 2));
                    }
                    else
                    {
                        features.Add(new CatBottleneck(base_channels * (long)Math.Pow(2, i + 2), base_channels * (long)Math.Pow(2, i + 2), block_num: block_num, stride: 1));
                    }
                }
            }
            return features;
        }

        public override List<Tensor> forward(Tensor input)
        {
            List<Tensor> out_feats = new List<Tensor>();
            Tensor x = input;
            x = features[0].forward(x);
            out_feats.Add(x);
            x = features[1].forward(x);
            out_feats.Add(x);
            for (int i = 2; i < 2 + _layers[0]; i++)
            {
                x = features[i].forward(x);
            }
            out_feats.Add(x);
            for (int i = (int)(2 + _layers[0]); i < (int)(2 + _layers[0] + _layers[1]); i++)
            {
                x = features[i].forward(x);
            }
            out_feats.Add(x);
            for (int i = (int)(2 + _layers[0] + _layers[1]); i < (int)(2 + _layers.Sum()); i++)
            {
                x = features[i].forward(x);
            }
            out_feats.Add(x);
            return out_feats;
        }
    }

    public class MattingNet : Module<Tensor, Tensor>
    {
        private readonly STDCNet backbone;
        private readonly List<long> _backbone_channels;
        private readonly List<long> _dpp_index;
        private readonly DoublePyramidPoolModule dpp;
        private readonly MLFF mlff32x;
        private readonly MLFF mlff16x;
        private readonly MLFF mlff8x;
        private readonly MLFF mlff4x;
        private readonly MLFF mlff2x;
        private readonly MattingHead matting_head_mlff8x;
        private readonly MattingHead matting_head_mlff2x;

        public MattingNet(long dpp_mid_channel = 256, long dpp_output_channel = 256, long out_channels = 1, long dpp_mlp_ratios = 2, long dpp_attn_ratio = 2, long head_channel = 8) : base("MattingNet")
        {
            List<long> dpp_index = new List<long>(new long[] { 1, 2, 3, 4 });
            List<long> dpp_bin_sizes = new List<long>(new long[] { 2, 4, 6 });
            List<long> decoder_channels = new List<long>(new long[] { 128, 96, 64, 32, 16 });
            backbone = new STDCNet(new List<long>(new long[] { 2, 2, 2 }));
            _backbone_channels = backbone._feat_channels;
            _dpp_index = dpp_index;
            long sum = 0;
            foreach (var index in dpp_index)
            {
                sum += _backbone_channels[(int)index];
            }
            dpp = new DoublePyramidPoolModule(2, sum, dpp_mid_channel, dpp_output_channel, dpp_bin_sizes, dpp_mlp_ratios, dpp_attn_ratio);
            List<long> mlff32xin = new List<long>(new long[] { _backbone_channels[_backbone_channels.Count - 1], dpp_output_channel });
            List<long> mlff32xout = new List<long>(new long[] { dpp_output_channel, dpp_output_channel });
            mlff32x = new MLFF(mlff32xin, mlff32xout, decoder_channels[0]);
            List<long> mlff16xin = new List<long>(new long[] { _backbone_channels[_backbone_channels.Count - 2], decoder_channels[0], dpp_output_channel });
            List<long> mlff16xout = new List<long>(new long[] { decoder_channels[0], decoder_channels[0], decoder_channels[0] });
            mlff16x = new MLFF(mlff16xin, mlff16xout, decoder_channels[1]);
            List<long> mlff8xin = new List<long>(new long[] { _backbone_channels[_backbone_channels.Count - 3], decoder_channels[1], dpp_output_channel });
            List<long> mlff8xout = new List<long>(new long[] { decoder_channels[1], decoder_channels[1], decoder_channels[1] });
            mlff8x = new MLFF(mlff8xin, mlff8xout, decoder_channels[2]);
            List<long> mlff4xin = new List<long>(new long[] { _backbone_channels[_backbone_channels.Count - 4], decoder_channels[2], 3 });
            List<long> mlff4xout = new List<long>(new long[] { decoder_channels[2], decoder_channels[2], 3 });
            mlff4x = new MLFF(mlff4xin, mlff4xout, decoder_channels[3]);
            List<long> mlff2xin = new List<long>(new long[] { _backbone_channels[_backbone_channels.Count - 5], decoder_channels[3], 3 });
            List<long> mlff2xout = new List<long>(new long[] { decoder_channels[3], decoder_channels[3], 3 });
            mlff2x = new MLFF(mlff2xin, mlff2xout, decoder_channels[4]);
            matting_head_mlff8x = new MattingHead(decoder_channels[2], 32);
            matting_head_mlff2x = new MattingHead(decoder_channels[4] + 3, head_channel, mid_num: 2);

            RegisterComponents();
        }
        public override Tensor forward(Tensor input)
        {
            using var _ = NewDisposeScope();
            var img = input;
            var input_shape = img.shape;
            var feats_backbone = backbone.forward(img);
            List<Tensor> dppList = new List<Tensor>();
            foreach (var index in _dpp_index)
            {
                dppList.Add(feats_backbone[(int)index]);
            }
            var x = dpp.forward(dppList);

            var dpp_out = x;
            List<Tensor> input_32x = new List<Tensor>();
            input_32x.Add(feats_backbone[feats_backbone.Count - 1]);
            input_32x.Add(x);
            long[] input_32xshape = { feats_backbone[feats_backbone.Count - 1].shape[2], feats_backbone[feats_backbone.Count - 1].shape[3] };

            x = mlff32x.forward(input_32x, input_32xshape);

            List<Tensor> input_16x = new List<Tensor>();
            input_16x.Add(feats_backbone[feats_backbone.Count - 2]);
            input_16x.Add(x);
            input_16x.Add(dpp_out);
            long[] input_16xshape = { feats_backbone[feats_backbone.Count - 2].shape[2], feats_backbone[feats_backbone.Count - 2].shape[3] };
            x = mlff16x.forward(input_16x, input_16xshape);

            List<Tensor> input_8x = new List<Tensor>();
            input_8x.Add(feats_backbone[feats_backbone.Count - 3]);
            input_8x.Add(x);
            input_8x.Add(dpp_out);
            long[] input_8xshape = { feats_backbone[feats_backbone.Count - 3].shape[2], feats_backbone[feats_backbone.Count - 3].shape[3] };
            x = mlff8x.forward(input_8x, input_8xshape);

            List<Tensor> input_4x = new List<Tensor>();
            long[] input_4xshape = { feats_backbone[feats_backbone.Count - 4].shape[2], feats_backbone[feats_backbone.Count - 4].shape[3] };
            input_4x.Add(feats_backbone[feats_backbone.Count - 4]);
            input_4x.Add(x);
            input_4x.Add(torch.nn.functional.interpolate(img, input_4xshape, mode: InterpolationMode.Area));
            x = mlff4x.forward(input_4x, input_4xshape);

            List<Tensor> input_2x = new List<Tensor>();
            long[] input_2xshape = { feats_backbone[feats_backbone.Count - 5].shape[2], feats_backbone[feats_backbone.Count - 5].shape[3] };
            input_2x.Add(feats_backbone[feats_backbone.Count - 5]);
            input_2x.Add(x);
            input_2x.Add(torch.nn.functional.interpolate(img, input_2xshape, mode: InterpolationMode.Area));
            x = mlff2x.forward(input_2x, input_2xshape);

            x = torch.nn.functional.interpolate(x, new long[] { input_shape[input_shape.Length - 2], input_shape[input_shape.Length - 1] }, mode: InterpolationMode.Bilinear, align_corners: false);
            List<Tensor> cat = new List<Tensor>();
            cat.Add(x);
            cat.Add(img);
            x = torch.cat(cat, 1);
            var alpha = matting_head_mlff2x.forward(x);
            return alpha.MoveToOuterDisposeScope();
        }
    }
}
