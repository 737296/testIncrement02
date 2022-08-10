from PySide6.QtCore import *  # type: ignore
from PySide6.QtGui import *  # type: ignore
from PySide6.QtWidgets import *  # type: ignore


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(542, 441)
        self.widget = QWidget(Form)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(150, 90, 258, 225))
        self.verticalLayout = QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.pushButton = QPushButton(self.widget)
        self.pushButton.setObjectName(u"pushButton")

        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton2 = QPushButton(self.widget)
        self.pushButton2.setObjectName(u"pushButton")

        self.verticalLayout.addWidget(self.pushButton2)

        self.label1 = QLabel(self.widget)
        self.label1.setObjectName(u"label")

        self.verticalLayout.addWidget(self.label1, 0, Qt.AlignHCenter)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.pushButton.setText(QCoreApplication.translate("Form", u"\u6587\u4ef6\u9009\u62e9", None))
#if QT_CONFIG(whatsthis)
        self.label1.setWhatsThis(QCoreApplication.translate("Form", u"<html><head/><body><p>\u8fd9\u662f\u4ec0\u4e48</p></body></html>", None))
#endif // QT_CONFIG(whatsthis)
        self.label1.setText(QCoreApplication.translate("Form", u"\u8fd9\u91cc\u8f93\u51fa\u6587\u4ef6\u8def\u5f84", None))
    # retranslateUi
