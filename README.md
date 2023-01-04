# Logistic Regression

## 1. Tổng quan

### 1.1 Giới thiệu

+ Logistic Regression là một thuộc toán thuộc nhóm Supervised Learning sử dụng rất hiệu quả cho classification.
+ Logistic Regression (còn gọi là Logit Regression) thường được sử dụng để ước tính xác suất mà một mẫu thuộc về một lớp cụ thể (ví dụ xác suất một email có phải là spam hay không hay xác suất một giao dịch có phải là gian lận hay không, ...)
+ Nếu xác suất ước tính lớn hơn 50% thì mô hình dự đoán mẫu thuộc về lớp đó (được gọi là lớp positive, có nhãn "1"), hoặc ngược lại dự đoán rằng nó không thuộc về lớp đó (được gọi là lớp negative, có nhãn "0") => tạo ra một phân loại nhị phân (Binary Classifier).
+ Binary Classifier (Phân loại nhị phân): outcome chỉ có 1 và 0 (hay đúng và sai) mặc dù trong tên có Regression, có thể gọi là Conditional Class Probabilities.

### 1.2 Các ứng dụng

+ Tiếp thị (Marketing): Dự đoán liệu một khách hàng cụ thể có mua sản phẩm bảo hiểm hay không?
+ Kinh doanh (Business): Dự đoán liệu khách hàng có ngưng sử dụng sản phẩm/dịch vụ không?
+ Banking (Ngân hàng): Dự đoán liệu khách hàng sẽ mặc định cho một khoản vay không?
+ Financial (Tài chính): Dự báo doanh nghiệp có gặp rủi ro phá sản/rủi ro tài chính hay không?

### 1.3 Lí do không áp dụng Linear Regression với bài toán chỉ có hai outcome

+ Khi response variable chỉ có hai giá trị cụ thể, ta mong muốn có một mô hình dự đoán giá trị là 0 hoặc 1 hoặc là một điểm xác suất nằm trong khoảng từ 0 đến 1. Do đó, Linear Regression không có khả năng này.

### 1.4 Thuật toán

+ Là một mô hình hồi quy, trong đó response variable (biến phụ thuộc) có các giá trị phân loại như TRUE/FALSE hoặc 1/0. Nó đo lường xác suất của một phản ứng nhị phân như giá trị của response variable dựa trên phương trình toán học liên quan tới response variable và các biến predictor variable.
+ Phương trình toán học (sigmod): $S(z)=\frac{1}{1 + e^{-z}}$  
+ Logistic Regression không phải là một thuật toán phân loại của riêng nó. Nó là một thuật toán phân loại kết hợp với một quy tắc quyết định tạo ra sự phân đôi các xác suất dự đoán của kết quả. Logistic Regression là một regression model vì nó ước tính xác suất thành viên của lớp như một hàm đa tuyến (multilinear function) của các feature.
+ Có thể nói: bất kỳ quá trình nào cố gắng tìm kiếm mối quan hệ giữa các biến được gọi là "regression". 
    + Logistic regression là "regression" vì nó tìm mối quan hệ giữa các biến. 
    + Logistic regression là "logistic" vì nó sử dụng chức năng logistic như một chức năng liên kết.
    
Một ví dụ đơn giản về Logistic Regression là: Lượng Calo, thời tiết và tuổi tác có ảnh hưởng gì đến nguy cơ bị đau tim không?

## 2. Ưu/khuyết điểm

### 2.1 Ưu điểm

+ Đây là một mô hình phân lớp hoạt động hiệu quả, dễ triển khai
+ Dễ dàng mở rộng cho bài toán target có nhiều hơn hai loại
+ Huấn luyện nhanh
+ Độ chính xác cao cho nhiều tập dữ liệu đơn giản
+ Có thể giải thích các hệ số mô hình cũng như các chỉ số về tầm quan trọng của tính năng

### 2.2 Khuyết điểm

+ Ranh giới quyết định tuyến tính: Mô hình này chỉ phù hợp với loại dữ liệu mà hai class là gần với linearly separable. Một kiểu dữ liệu mà Logistic Regression không làm việc được là khi dữ liệu mà một class chứa các điểm nằm trong một vòng tròn, class kia chứa các điểm bên ngoài đường tròn đó.
+ Một hạn chế nữa của Logistic Regression là nó yêu cầu các điểm dữ liệu được tạo ra một cách độc lập với nhau. Trên thực tế, các điểm dữ liệu có thể bị ảnh hưởng bởi nhau.

## 3. Xây dựng Logistic Regression

Dùng `sklearn.linear_model.LogisticRegression`

### Các bước thực hiện

+ Chọn model sẽ sử dụng là: Logistic Regression
+ Đọc dữ liệu, tiền xử lý dữ liệu
+ Tạo một tập dữ liệu features (inputs) và một tập target (output) chứa các nhãn cho các mẫu => chia dữ liệu thành hai bộ train và test.
+ Xây dựng model với training data
+ Đánh giá độ chính xác
+ Trực quan hoá kết quả
+ Dự đoán mới
