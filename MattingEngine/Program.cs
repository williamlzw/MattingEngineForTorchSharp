using OpenCvSharp;
using HumanSegEngine;

static class Program
{
    public static void Main()
    {
        test_mattingengine();
    }

    public static void test_mattingengine()
    {
        MattingModelConfig config = new MattingModelConfig();
       
        config.mattingModelPath = "data\\matting.dat";
        MattingParameter parameter = new MattingParameter();
        parameter.UseGpu = false;
        var engine = new MattingEngine(config, parameter);
        string path = "data\\6.jpg";
        var time0 = Environment.TickCount;
        var ret = engine.DetectHuman(path);
        var time1 = Environment.TickCount;
        Console.WriteLine((time1 - time0));
        Cv2.ImWrite("reg_result.png", ret);
    }
}
