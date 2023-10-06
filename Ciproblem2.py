import numpy as np

# อ่านไฟล์ cross.txt
def read_cross_file(filename):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            label = lines[i].strip()
            x, y = map(float, lines[i + 1].split())
            i += 2
            label_data = (label, (x, y),)
            label_data += tuple(map(int, lines[i].split()))
            i += 1
            data.append(label_data)
    return data

def split_and_shuffle_data(data, train_ratio=0.9):
    np.random.shuffle(data)
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# แปลงข้อมูลเป็นรูปแบบที่เหมาะสำหรับ Neural Network
def prepare_data(data):
    inputdata = np.array([item[1] for item in data])
    outputdata = np.array([item[2:] for item in data])
    return inputdata, outputdata

# ฟังก์ชันคำนวณ sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ฟังก์ชันคำนวณ sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

#อัพเดทค่าออกจาก hidden note และ output note
def forward_propagation(input_data):
    hidden = sigmoid(np.dot(w_input_to_hidden, input_data.T) + b_hidden)
    output = sigmoid(np.dot(w_hidden_to_output, hidden) + b_output)

    return hidden, output

#อัพเดทค่า weight ระหว่าง input เข้า hidden note และ bias เข้า hidden note
def update_input_hidden_layer_weights(input_data, hidden_gradient, learning_rate, momentum_rate):
    global w_input_to_hidden, b_hidden, v_w_input_hidden, v_b_hidden
    v_w_input_hidden = (momentum_rate * v_w_input_hidden) + (learning_rate * np.dot(hidden_gradient, input_data) / len(input_data))
    w_input_to_hidden += v_w_input_hidden

    v_b_hidden = (momentum_rate * v_b_hidden) + (learning_rate * np.mean(hidden_gradient, axis=1, keepdims=True))
    b_hidden += v_b_hidden
    
#อัพเดทค่า weight ระหว่าง hidden note เข้า output note และ bias เข้า output note
def update_hidden_output_layer_weights(hidden, output_gradient, learning_rate, momentum_rate):
    global w_hidden_to_output, b_output, v_w_hidden_output, v_b_output
    v_w_hidden_output = (momentum_rate * v_w_hidden_output) + (learning_rate * np.dot(output_gradient, hidden.T) / len(hidden))
    w_hidden_to_output += v_w_hidden_output

    v_b_output = (momentum_rate * v_b_output) + (learning_rate * np.mean(output_gradient, axis=1, keepdims=True))
    b_output += v_b_output
    
#นำชุดข้อมูลมาเทรน 
def train_custom_neural_network(inputdata, outputdata,Target_Epochs,Mean_Squared_Error, learning_rate, momentum_rate):
       
    for epochs in range(Target_Epochs):
        
        hidden, output = forward_propagation(inputdata) 

        output_error = outputdata - output.T
        output_gradient = output_error.T * sigmoid_derivative(output)
        update_hidden_output_layer_weights(hidden, output_gradient, learning_rate, momentum_rate)
    
        hidden_error = np.dot(w_hidden_to_output.T, output_gradient)
        hidden_gradient = hidden_error * sigmoid_derivative(hidden)
        update_input_hidden_layer_weights(inputdata, hidden_gradient, learning_rate, momentum_rate) 
        
        error = np.mean(output_error**2, axis=0)  # คำนวณ Output Error เฉลี่ยสำหรับแต่ละคลาส
        if epochs % 10000 == 0:
            print(f"Epoch loop: {epochs+10000}, Error: {error}")
            
        # เมื่อค่า error สำหรับแต่ละคลาสต่ำกว่าหรือเท่ากับ MSE ให้หยุดการฝึก
        if np.all(error <= Mean_Squared_Error):
            break 

#คำนวณค่า Accuracy เป็นเปอร์เซ็นต์จากค่า True Positive (TP), True Negative (TN), False Positive (FP), และ False Negative (FN)
def calculate_accuracy(TP, TN, FP, FN):
    total_predictions = TP + TN + FP + FN
    if total_predictions == 0:
        return 0.0
    accuracy_percentage = ((TP + TN) / total_predictions) * 100
    return accuracy_percentage
 
################################################# Main ####################################################################      
filename = 'cross.txt'
cross_data = read_cross_file(filename)
train, test = split_and_shuffle_data(cross_data)

input_size = 2
hidden_size = 2 # สามารถกำหนดเองได้
output_size = 2
        
#initialize weight แตกต่างกัน โดย สร้างตัวแปร array สุ่มค่า weight และ bias เริ่มต้น รวมถึง สร้างตัวแปร array สำหรับสุ่มค่าสำหรับอัปเดทค่า weight และ bias 
#weight ระหว่าง input note เข้า hidden note
w_input_to_hidden = np.random.randn(hidden_size, input_size)
v_w_input_hidden = np.random.randn(hidden_size, input_size)

#weight ระหว่าง hidden note เข้า output note
w_hidden_to_output = np.random.randn(output_size, hidden_size)
v_w_hidden_output = np.random.randn(output_size, hidden_size)
    
#bias เข้า hidden note 
b_hidden = np.random.randn(hidden_size, 1)
v_b_hidden = np.random.randn(hidden_size, 1)
    
#bias เข้า  note 
b_output = np.random.randn(output_size, 1)
v_b_output = np.random.randn(output_size, 1)


# ปรับ learning_rates และ momentum_rates ตามที่ต้องการ    
learning_rates = [ 0.1]
momentum_rates = [ 0.01]
    
for lr in learning_rates:
    for momentum in momentum_rates: 
        print(f"Training with learning rate = {lr} and momentum = {momentum}")
        
        # แยกชุดข้อมูล train 
        inputtrain, outputtrain = prepare_data(train)
            
        #นำข้อมูล train มาฝึกโดยสามารถกำหนด จำนวน epoch และ ค่าคลาดเคลื่อนเฉลี่ย MSE ที่ต้องการได้ 
        train_custom_neural_network(inputtrain,outputtrain, 50000, 0.00001, lr, momentum)
            
        # แยกชุดข้อมูล test 
        inputtest, outputtest = prepare_data(test)
        
        #เปลี่ยนชื่อให้ง่ายต่อความเข้าใจ
        Actual = outputtest
        
        #นำข้อมูล test เข้าไปหาค่า Predict output จากการเทรน
        x,Predict = forward_propagation(inputtest)
        
        #แปลงข้อมูลจาก (2,20)->(20,2)
        Predict = np.transpose(Predict)   
        
        # สร้าง confusion matrix
        # กำหนดค่า Threshold เพื่อแปลงค่าความน่าจะเป็นเป็นค่าทางด้านไปหรือค่าทางด้านใน
        threshold = 0.5
        predicted = (Predict[:, 1] > threshold).astype(int)

        # คำนวณ Confusion Matrix
        confusion_matrix = np.zeros((2, 2), dtype=int)
        for i in range(2):
            for j in range(2):
                confusion_matrix[i, j] = np.sum((Actual[:, i] == 1) & (predicted == j))

        # คำนวณ True Positive (TP), True Negative (TN), False Positive (FP), และ False Negative (FN)
        TP = confusion_matrix[1, 1]
        TN = confusion_matrix[0, 0]
        FP = confusion_matrix[0, 1]
        FN = confusion_matrix[1, 0]
        Accuracy = calculate_accuracy(TP, TN, FP, FN)
        
        print("Confusion Matrix:")
        print(confusion_matrix)
        print("True Positive (TP):", TP)
        print("True Negative (TN):", TN)
        print("False Positive (FP):", FP)
        print("False Negative (FN):", FN)
        print(f"************Accuracy = {Accuracy} % **************")
 

