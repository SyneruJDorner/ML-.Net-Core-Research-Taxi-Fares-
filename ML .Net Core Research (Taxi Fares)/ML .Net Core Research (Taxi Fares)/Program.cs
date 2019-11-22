using System;
using Microsoft.ML;
using ML__Net_Core_Research__Taxi_Fares_ML.Model;

namespace ML_.Net_Core_Research__Taxi_Fares_
{
    class Program
    {
        static void Main(string[] args)
        {
            //Load the Model
            MLContext mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load("MLModel.zip", out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            //Use the code below to add ionput data
            ModelInput input = new ModelInput()// Create sample data
            {
                Vendor_id = "CMT",
                Rate_code = 1,
                Passenger_count = 1,
                Trip_distance = 3.8f,
                Payment_type = "CRD"
            };

            // Make prediction
            ModelOutput prediction = ConsumeModel.Predict(input);

            // Print Prediction
            Console.WriteLine($"Predicted Fare: {prediction.Score}");
            Console.ReadKey();
        }
    }
}
