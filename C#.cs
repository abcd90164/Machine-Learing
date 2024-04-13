using System;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Drawing;

class Program
{
    static void Main()
    {
        // Load the image
        string imagePath = "Donald_Trump.jpg";
        Mat image = CvInvoke.Imread(imagePath, ImreadModes.Color);

        // Convert to grayscale
        Mat grayImage = new Mat();
        CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);

        // Load the cascade classifier
        string faceCascadePath = "face_id.xml";
        CascadeClassifier faceCascade = new CascadeClassifier(faceCascadePath);

        // Detect faces
        Rectangle[] faces = faceCascade.DetectMultiScale(
            grayImage, 
            1.1, 
            10, 
            new Size(20, 20));

        // Draw rectangles around each face
        foreach (var face in faces)
        {
            CvInvoke.Rectangle(image, face, new Bgr(Color.Green).MCvScalar, 2);
        }

        // Display the image using a window
        using (var window = new Emgu.CV.UI.ImageViewer())
        {
            window.Image = image;
            window.ShowDialog();
        }
    }
}
