# VBD-CL
* Đây là mã nguồn cài đặt phương pháp VBD-CL. 
* Ý nghĩa các file cài đặt chính:
-) conv_net.py: mô hình mạng CNN cho tập Split CIFAR100 và Split CIFAR10-100.
-) omniglot_conv_net.py: mô hình mạng CNN cho tập Split Omniglot.
-) model.py: mô hình mạng MLP cho tập Split MNIST và Permuted MNIST.
-) layers_VBD: cài đặt tầng nhiễu của Dropout.
-) data.py: tạo dữ liệu cho các kịch bản Học liên tục.
* Chú ý: 
-) Chạy file test_vbd_sgd.py để thực hiện các thử nghiệm. Lưu ý các tham số trong đó cần được xác định thủ công Mã nguồn vẫn đang trong quá trình hoàn thiện.
-) Đường dẫn folder dataset trong file data.py cần được thay đổi cho phù hợp.
-) Nếu phát hiện sai sót trong mã nguồn, vui lòng liên hệ tới địa chỉ mail sau đây:
tunglamlqddb@gmail.com
