using System.Runtime.InteropServices;
using OpenCvSharp;
using TorchSharp;


namespace ProcessHelp
{
    public enum TransType
    {
        Resize,
        Padding
    }
    public class ShapeInfo
    {
        public ShapeInfo(int width, int height, TransType type)
        {
            Width = width;
            Height = height;
            Type = type;
        }
        public int Width;
        public int Height;
        public TransType Type;
    }
    public class TransDataInfo
    {
        public Mat img = null;
        public List<ShapeInfo> trans = new List<ShapeInfo>();
    }
    public static class PreProcess
    {
        public class LoadImagesOp
        {
            private bool _toRGB;
            public LoadImagesOp(bool toRGB = true)
            {
                _toRGB = toRGB;
            }
            public TransDataInfo Call(string path)
            {
                TransDataInfo data = new TransDataInfo();
                data.img = Cv2.ImRead(path);
                if(_toRGB)
                {
                    Cv2.CvtColor(data.img, data.img, ColorConversionCodes.BGR2RGB);
                }
                return data;
            }
        }

        public class ToTensorOp
        {
            public ToTensorOp()
            {

            }
            public torch.Tensor Call(TransDataInfo data, bool RGB = true)
            {
                if(RGB == false)
                {
                    Cv2.CvtColor(data.img, data.img, ColorConversionCodes.BGR2RGB);
                }
                byte[] dataByte = new byte[data.img.Total() * 3];
                Marshal.Copy(data.img.Data, dataByte, 0, dataByte.Length);
                return torch.tensor(dataByte, torch.ScalarType.Byte).reshape(data.img.Height, data.img.Width, 3).permute(2, 0, 1);
            }
        }
        private static Mat ResizeLong(Mat im, int longSize = 224)
        {
            var value = Math.Max(im.Height, im.Width);
            double scale = (double)longSize / value;
            int resizedWidth = (int)Math.Round(im.Width * scale);
            int resizedHeight = (int)Math.Round(im.Height * scale);
            Cv2.Resize(im, im, new OpenCvSharp.Size(resizedWidth, resizedHeight), interpolation: InterpolationFlags.Linear);
            return im;
        }

        public static Mat ResizeShort(Mat im, int shortSize = 224)
        {
            var value = Math.Min(im.Height, im.Width);
            double scale = (double)shortSize / value;
            int resizedWidth = (int)Math.Round(im.Width * scale);
            int resizedHeight = (int)Math.Round(im.Height * scale);
            Cv2.Resize(im, im, new OpenCvSharp.Size(resizedWidth, resizedHeight), interpolation: InterpolationFlags.Linear);
            return im;
        }

        public static Mat Resize(Mat im, int targetWidth = 608, int targetHeight = 608)
        {
            Cv2.Resize(im, im, new OpenCvSharp.Size(targetWidth, targetHeight), interpolation: InterpolationFlags.Linear);
            return im;
        }

        public class LimitShortOp
        {
            private int _maxShort;
            private int _minShort;
            public LimitShortOp(int maxShort = 0, int minShort = 0)
            {
                _minShort = minShort;
                _maxShort = maxShort;
            }

            public TransDataInfo Call(TransDataInfo data)
            {
                var w = data.img.Width;
                var h = data.img.Height;
                var shortEdge = Math.Min(w, h);
                var target = shortEdge;
                if(_maxShort != 0 && shortEdge > _maxShort)
                {
                    target = _maxShort;
                }
                else if(_minShort !=0 && shortEdge < _minShort)
                {
                    target = _minShort;
                }
                data.trans.Add(new ShapeInfo(data.img.Width, data.img.Height, TransType.Resize));
                if(target != shortEdge)
                {
                    data.img = ResizeShort(data.img, target);
                }
                return data;
            }
        }

        public class LimitLongOp
        {
            private int _maxShort;
            private int _minShort;
            public LimitLongOp(int maxShort = 0, int minShort = 0)
            {
                _minShort = minShort;
                _maxShort = maxShort;
            }

            public TransDataInfo Call(TransDataInfo data)
            {
                var w = data.img.Width;
                var h = data.img.Height;
                var longEdge = Math.Max(w, h);
                var target = longEdge;
                if (_maxShort != 0 && longEdge > _maxShort)
                {
                    target = _maxShort;
                }
                else if (_minShort != 0 && longEdge < _minShort)
                {
                    target = _minShort;
                }
                data.trans.Add(new ShapeInfo(data.img.Width, data.img.Height, TransType.Resize));
                if (target != longEdge)
                {
                    data.img = ResizeLong(data.img, target);
                }
                return data;
            }
        }

        public class ResizeToIntMultOp
        {
            private int _multInt;
            public ResizeToIntMultOp(int multInt = 32)
            {
                _multInt = multInt;
            }

            public TransDataInfo Call(TransDataInfo data)
            {
                data.trans.Add(new ShapeInfo(data.img.Width, data.img.Height, TransType.Resize));
                var w = data.img.Width;
                var h = data.img.Height;
                var rw = w - w % _multInt;
                var rh = h - h % _multInt;
                data.img = Resize(data.img, rw, rh);
                return data;
            }
        }

        public static Mat Normalize(Mat im, double[] mean, double[] std)
        {
            var bgrChannels = Cv2.Split(im);
            for(int index = 0; index < bgrChannels.Length; index++)
            {
                bgrChannels[index].ConvertTo(bgrChannels[index], MatType.CV_32FC1, 1 / std[index], -mean[index] / std[index]);
            }
            Mat dst = im;
            Cv2.Merge(bgrChannels, dst);
            return dst;
        }

        public class NormalizeOp
        {
            private double[] _mean;
            private double[] _std;
            public NormalizeOp(double[] mean , double[] std)
            {
                _mean = mean;
                _std = std;
            }

            public TransDataInfo Call(TransDataInfo data)
            {
                data.img = Normalize(data.img, _mean, _std);
                return data;
            }
        }

        public class PaddingOp
        {
            private int _targetWidth;
            private int _targetHeight;
            private OpenCvSharp.Scalar _paddingValue;
            public PaddingOp(int targetWidth, int targetHeight, OpenCvSharp.Scalar paddingValue)
            {
                _targetWidth = targetWidth;
                _targetHeight = targetHeight;
                _paddingValue = paddingValue;
            }

            public TransDataInfo Call(TransDataInfo data)
            {
                var h = data.img.Height;
                var w = data.img.Width;
                var padHeight = Math.Max(0, _targetHeight - h);
                var padWidth = Math.Max(0, _targetWidth - w);
                data.trans.Add(new ShapeInfo(data.img.Width, data.img.Height, TransType.Padding));
                if (padWidth == 0 && padHeight == 0)
                {
                    return data;
                }
                else
                {
                    Cv2.CopyMakeBorder(data.img, data.img, 0, padHeight, 0, padWidth, BorderTypes.Constant, _paddingValue);
                }
                return data;
            }
        }
    }

    public static class PostProcess
    {
        public static torch.Tensor ReverseTransformTorch(torch.Tensor alpha, TransDataInfo info)
        {
           var trans = info.trans;
           trans.Reverse();
           foreach (var item in trans)
           {
                if(item.Type == TransType.Resize)
                {
                    var h = item.Height;
                    var w = item.Width;
                    alpha = torch.nn.functional.interpolate(alpha, new long[] { h, w }, mode: torch.InterpolationMode.Bilinear, align_corners:false);
                }
                else if(item.Type == TransType.Padding)
                {
                    var h = item.Height;
                    var w = item.Width;
                    alpha = alpha[torch.TensorIndex.Colon, torch.TensorIndex.Colon, torch.TensorIndex.Slice(0, h), torch.TensorIndex.Slice(0, w)];
                }
           }
           return alpha;
        }

        public static Mat TensorToMat(torch.Tensor tensor, MatType type)
        {
            var height = tensor.shape[0];
            var width = tensor.shape[1];
            var channel = tensor.shape[2];
            Mat mat = null;
            if (channel == 3)
            {
                mat = new Mat((int)height, (int)width, type);
            }
            else if (channel == 1)
            {
                mat = new Mat((int)height, (int)width, type);
            }
            var access = tensor.data<byte>();
            var data = access.ToArray();
            Marshal.Copy(data, 0, mat.Data, data.Length);
            return mat;
        }
    }
}
