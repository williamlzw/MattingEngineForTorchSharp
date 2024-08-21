using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MattingNetHelp;
using TorchSharp;
using ProcessHelp;
using System.IO;
using OpenCvSharp;

namespace HumanSegEngine
{
    public class MattingParameter
    {
        public bool UseGpu = false;
        public int GpuId = 0;
        public double threshold = 0.4;
        public double nms_iou = 0.5;
    }

    public class MattingModelConfig
    {
        public string mattingModelPath;
    }
    public class MattingEngine
    {
        private MattingNet m_mattingModel;
        private torch.Device m_device;
        private MattingParameter m_parameter;
        private torchvision.ITransform m_normalizeDetOperator;
        private torchvision.ITransform m_transformOperator;
        private PreProcess.LoadImagesOp m_loadOp;
        private PreProcess.LimitShortOp m_limitOp;
        private PreProcess.ResizeToIntMultOp m_resizeIntOp;
        private PreProcess.PaddingOp m_paddingOp;
        private PreProcess.ToTensorOp m_totensorOp;

        public MattingEngine(MattingModelConfig config, MattingParameter parameter)
        {
            if (!File.Exists(config.mattingModelPath)) throw new FileNotFoundException(config.mattingModelPath);
            m_parameter = parameter;
            var description = $"cpu";
            if (parameter.UseGpu)
            {
                description = "cuda:0";
            }
            m_device = torch.device(description);
            m_mattingModel = new MattingNet();
            m_mattingModel.load(config.mattingModelPath);
            m_mattingModel.eval();
            m_mattingModel.to(m_device);
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
            m_normalizeDetOperator = torchvision.transforms.Normalize(new double[] { 0.5, 0.5, 0.5 }, new double[] { 0.5, 0.5, 0.5 });
            m_transformOperator = torchvision.transforms.ConvertImageDType(torch.ScalarType.Float32);

            m_loadOp = new PreProcess.LoadImagesOp();
            m_limitOp = new PreProcess.LimitShortOp(512);
            m_resizeIntOp = new PreProcess.ResizeToIntMultOp(32);
            m_paddingOp = new PreProcess.PaddingOp(512, 512, new OpenCvSharp.Scalar(127.5, 127.5, 127.5));
            m_totensorOp = new PreProcess.ToTensorOp();
        }

        public Mat DetectHuman(string path)
        {
            var info = m_loadOp.Call(path);
            var ori = Cv2.ImRead(path);
            info = m_limitOp.Call(info);
            info = m_resizeIntOp.Call(info);
            info = m_paddingOp.Call(info);
            var image = m_totensorOp.Call(info);
            image = m_normalizeDetOperator.call(m_transformOperator.call(image).unsqueeze(0));
            image = image.to(m_device);
            var alphaPred = m_mattingModel.forward(image);
            alphaPred = PostProcess.ReverseTransformTorch(alphaPred, info);
            alphaPred = alphaPred.squeeze().unsqueeze(-1);
            alphaPred = (alphaPred * 255).to(torch.uint8);
            alphaPred = alphaPred.detach().cpu();
            var alpha = PostProcess.TensorToMat(alphaPred, OpenCvSharp.MatType.CV_8UC1);
            Mat rgba = new Mat(alpha.Height, alpha.Width, MatType.CV_8UC3);
            Cv2.Merge(new Mat[] { ori, alpha }, rgba);
            return rgba;
        }
    }
}
