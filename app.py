import sys
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QCheckBox, QLineEdit, QMessageBox, QRadioButton
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import Qt, QTimer, QRect, QSize
import os
from time import sleep
import matplotlib.pyplot as plt

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_stage = None
    
    def initUI(self):
        self.setWindowTitle('Network Training App')
        self.setGeometry(100, 100, 800, 600)

        # Create widgets
        self.stage_label = QLabel('Select Stage:')
        # Create three radio buttons for stages
        self.stage_buttons = []
        for stage in range(1, 4):
            radio_button = QRadioButton(f'Stage {stage}')
            self.stage_buttons.append(radio_button)        

        buttons_layout = QHBoxLayout()
        for button in self.stage_buttons:
            buttons_layout.addWidget(button)
        # Set the default checked state for the first radio button
        self.stage_buttons[0].setChecked(True)
        self.selected_stage = 1  

        self.pretrained_checkbox = QCheckBox('Is Pretrained Unlock')
        self.inherit_checkpoint_checkbox = QCheckBox('Is Checkpoint Inheritted')

        self.train_button = QPushButton('Start Training')
        self.train_new_button = QPushButton('Train New Network')

        self.max_steps_label = QLabel('Max Steps When Not Stopped:')
        self.max_steps_input = QLineEdit()
        self.max_steps_input.setText(str(100000))

        self.learning_rate_label = QLabel('Learning rate: (If you check the inherit checkpoint, you can set learning rate <= 0 to inherit the lr from the stored adam optimizer.)')
        self.learning_rate_input = QLineEdit()
        self.learning_rate_input.setText(str(0.000001))

        # Image display area
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap('not_ready.png')
        pixmap = pixmap.scaled(100, 100)
        self.image_label.setPixmap(pixmap)

        # GIF display
        '''self.gif_label = QLabel(self)
        self.gif_label.setAlignment(Qt.AlignCenter)
        #self.gif_label.setGeometry(QRect(0, 0, 200, 200))
        #self.gif_label.setMinimumSize(QSize(640, 360))
        #self.gif_label.setMaximumSize(QSize(640, 360))
        self.gif_label.setScaledContents(True) 
        self.gif_label.setObjectName('/media/caotun8plus9/linux_drive/output.gif')
        self.movie = QMovie('/media/caotun8plus9/linux_drive/output.gif')
        #self.gif_label.setStyleSheet('position: absolute;left: 500;border: 3px solid green;padding: 10px;')
        self.gif_label.setMovie(self.movie)
        self.movie.start()'''
        

        # Display the integer from the file
        self.last_old_time_label = QLabel(f'# of Already Trained Steps (Updated per 100 secs): {self.read_last_old_time()}')
        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label) 
        layout.addWidget(self.stage_label)
        layout.addLayout(buttons_layout)
        layout.addWidget(self.pretrained_checkbox)
        layout.addWidget(self.inherit_checkpoint_checkbox)
        layout.addWidget(self.max_steps_label)
        layout.addWidget(self.max_steps_input)
        layout.addWidget(self.learning_rate_label)
        layout.addWidget(self.learning_rate_input)
        layout.addWidget(self.train_button)
        layout.addWidget(self.train_new_button)
        layout.addWidget(self.last_old_time_label)
        #layout.addWidget(self.gif_label)
        #self.gif_label.move(self.gif_label.x() + 500, self.gif_label.y() - 500)
        self.setLayout(layout)

        # Connect the button click event to the function
        for button in self.stage_buttons:
            button.toggled.connect(self.update_button_state)
        self.train_button.clicked.connect(self.toggle_train_network)
        self.train_new_button.clicked.connect(self.confirm_train_new_network)

        # Thread and flag for controlling training task
        self.task_thread = None
        self.is_training = False

        # Timer for checking and updating the image
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.check_image_modification)
        self.update_timer.start(1000000)  # Check every 1000 second (adjust as needed)

        # Timer for checking and update the last_old_time
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.updateLastOldTime)
        self.update_timer.start(10000)  # Check every 10 second (adjust as needed)

        # Initialize the image path and modification time
        self.image_path = 'reward_plot.png'  # Replace with the path to your image
        self.time_path = 'last_old_time.txt'
        self.oldTime_mtime = 0

    def updateLastOldTime(self):
        self.last_old_time_label.setText(f'# of Already Trained Steps (Updated per 100 secs): {self.read_last_old_time()}')

    def toggle_train_network(self):
        if not self.is_training:
            self.update_button_state()
            # Start training
            stage = self.selected_stage
            print("stage: ", stage)
            now_stage_file = open('now_stage.txt', 'r')
            now_stage = int(now_stage_file.readline())
            if now_stage > stage:
                self.wrong_stage_window(now_stage, stage)
                return
            is_pretrained_unlock = self.pretrained_checkbox.isChecked()
            is_inherit_checkpoint = self.inherit_checkpoint_checkbox.isChecked()
            try:
                max_steps = int(self.max_steps_input.text())
            except ValueError:
                print("Please type an int in the max step input box!")
                return
            try:
                lr = float(self.learning_rate_input.text())
            except:
                print("Please type an float in the learning rate input box!")
                return
            
            self.train_button.setText('Stop Training')
            self.is_training = True
            self.train_new_button.setEnabled(False)

            self.training_event = threading.Event()
            self.training_thread = threading.Thread(target=self.run_train_network, args=(stage, is_pretrained_unlock, max_steps, is_inherit_checkpoint, lr, self.training_event))
            self.training_thread.start()
            #self.run_train_network(stage, is_pretrained_unlock, max_steps, self.training_event)
        else:
            # Stop training
            self.train_new_button.setEnabled(True)
            self.is_training = False
            self.train_button.setText('Start Training')
            if self.training_thread:
                self.training_event.set()

    def run_train_network(self, stage, is_pretrained_unlock, is_inherit_checkpoint, lr, max_steps, event : threading.Event):
        self.check_image_modification()
        from deep_q_network import trainNetwork
        print(f"Training Network with stage={stage}, is_pretrained_unlock={is_pretrained_unlock}")
        trainNetwork(stage, is_pretrained_unlock, is_inherit_checkpoint, lr, max_steps, event)

    def confirm_train_new_network(self):
        # Show a confirmation dialog before proceeding with "Train New Network"
        confirmation = QMessageBox.question(self, 'Confirm Action', '阿如果你要訓練新的 Network你舊的模型就會被刪掉喔, 你要繼續嗎?',
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirmation == QMessageBox.Yes:
            self.train_new_network()  # Perform the action
        else:
            pass  # Do nothing if the user cancels

    def wrong_stage_window(self, now_stage, desired_stage):
        QMessageBox.information(None, 'Warning', f'你現在練到 stage{now_stage} 了, 結果你還要回去練 stage{desired_stage} ? 麻煩你選後面一點的 stage, 要不然就按 Train New Network重練一個')

    def wrong_learning_rate_window(self):
        QMessageBox.information(None, 'Warning', 'Learning rate should be larger than 0 unless you check the \"inherit checkpoint\".')

    def train_new_network(self):
        if os.path.exists("results.txt"):
            os.remove("results.txt")
        if os.path.exists("last_old_time.txt"):
            os.remove("last_old_time.txt")
        if os.path.exists("model/FlappyBird.h5"):
            os.remove("model/FlappyBird.h5")
        for i in range(3):
            if os.path.exists("Qvalues/Q"+str(i)+".txt"):
                os.remove("Qvalues/Q"+str(i)+".txt")
        now_stage_file = open('now_stage.txt', 'w')
        now_stage_file.write("1")
        now_stage_file.close()
        sleep(5)
        self.toggle_train_network()
    
    def update_button_state(self):
        # Update the state of the "Train New Network" button based on the selected stage
        self.selected_stage = None
        for index, button in enumerate(self.stage_buttons):
            if button.isChecked():
                self.selected_stage = index + 1
                break
        print(self.selected_stage, "\n\n\n\n")

        if self.selected_stage is not None and self.selected_stage != 1:
            self.train_new_button.setEnabled(False)
        else:
            self.train_new_button.setEnabled(True)

    def read_last_old_time(self):
        # Read the integer from the file 'last_old_time.txt'
        try:
            with open('last_old_time.txt', 'r') as file:
                return int(file.read())
        except (FileNotFoundError, ValueError):
            return 0

    def check_image_modification(self):
        # Check if the image file has been modified
        if os.path.exists(self.time_path):
            #current_mtime = os.path.getmtime(self.time_path)
            #if current_mtime != self.oldTime_mtime:
            self.drawReward()
        
    def drawReward(self):
        # Open the file for reading
        ctr = 0
        lines = []
        lines_sparse = []
        if os.path.exists('results.txt'):
            file = open('results.txt', 'r')
            if os.path.getsize('results.txt'):
            # Read all lines from the file and convert them to floats
                for line in file:
                  lines.append(float(line.strip()))
                  if ctr % 1000 == 0:
                    lines_sparse.append(float(line.strip()))
                  ctr += 1

                plt.plot(range(len(lines)), lines)
                plt.savefig("reward_plot.png")
                #plt.plot(range(len(lines_sparse)), lines_sparse)
                #plt.savefig("reward_plot_sparse.png")
                pixmap = QPixmap(self.image_path)
                self.image_label.setPixmap(pixmap)
            else:
                pixmap = QPixmap('not_ready.png')
                self.image_label.setPixmap(pixmap)
        else:
            pixmap = QPixmap('not_ready.png')
            self.image_label.setPixmap(pixmap)

    
def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
