using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace Monitor
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            DispatcherTimer timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromMilliseconds(1000 / 40);
            timer.Tick += Timer_Tick;
            timer.Start();


        }

        BitmapImage l;

        private void Timer_Tick(object sender, EventArgs e)
        {
            BitmapImage b = new BitmapImage(new Uri("http://localhost:8080/frame.png?t=" + new Random().Next()));
            b.DownloadCompleted += B_DownloadCompleted;
            l = b;
        }

        private void B_DownloadCompleted(object sender, EventArgs e)
        {
            myImgBrush.ImageSource = l;
        }
    }
}
