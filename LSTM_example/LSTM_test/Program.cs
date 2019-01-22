using System;
using TensorFlow;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace LSTM_test
{
    class Program
    {
        static private Dictionary<string, int> dict;
        static private Dictionary<int, string> reverse_dict;


        static void LoadDicts()
        {
            string json_dict = System.IO.File.ReadAllText(@"C:\Users\Ieva\Documents\Coding\IIB_Project\LSTM tutorial\LSTM_test\dictionary.json");
            dict = JsonConvert.DeserializeObject<Dictionary<string, int>>(json_dict);
            
            string json_rev = System.IO.File.ReadAllText(@"C:\Users\Ieva\Documents\Coding\IIB_Project\LSTM tutorial\LSTM_test\reverse_dictionary.json");
            reverse_dict = JsonConvert.DeserializeObject<Dictionary<int, string>>(json_rev);
        }

        static int GetArgMax(Single[,] matrix, int row_idx=0)
        {
            Single max_val = Single.NegativeInfinity;
            int arg_max = -1;
            Single current_val;

            for (int i=0; i<matrix.GetLength(1); i++)
            {
                current_val = matrix[row_idx, i];
                if (current_val > max_val)
                {
                    max_val = current_val;
                    arg_max = i;
                }
            }

            return arg_max;
        }

        static void Main(string[] args)
        {
            // Press Ctrl+F5 (or go to Debug > Start Without Debugging) to run your app.
            Console.WriteLine("Hello World!");
            LoadDicts();

            //load SavedModel
            TFGraph _tfGraph = new TFGraph();

            using (var tmpSess = new TFSession(_tfGraph))
            using (var tfSessionOptions = new TFSessionOptions())
            using (var metaGraphUnused = new TFBuffer())
            {
                //for some reason FromSavedModel is not static
                var session = tmpSess.FromSavedModel(tfSessionOptions, null, @"C:\Users\Ieva\Documents\Coding\IIB_Project\LSTM tutorial\LSTM_test\model_1548158314", new[] { "myTag" }, _tfGraph, metaGraphUnused);
                var runner = session.GetRunner();

                float[,,] symb_input = new float[,,] { 
                    { { dict["the"] } },
                    { { dict["mice"] } },
                    { { dict["had"] } }
                };
                var tensor = new TFTensor(symb_input);
                Console.WriteLine(_tfGraph["myInput"]);
                runner.AddInput(_tfGraph["myInput"][0], tensor);
                runner.Fetch(_tfGraph["myOutput"][0]);

                var output = runner.Run();
                var vecResults = output[0].GetValue();

                if (vecResults.GetType() == typeof(Single[,]))
                {
                    Single[,] result = (Single[,])vecResults;

                    Console.WriteLine(result.GetLength(0)); // Writes 5
                    Console.WriteLine(result.GetLength(1)); // Writes 10

                    int idx = GetArgMax(result);
                    Console.WriteLine(idx);
                }
                

            }


        }


    }


}
