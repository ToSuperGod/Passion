#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QLabel>
#include <QPainter>
#include <QPushButton>
#include <QDebug>
#include <QFileDialog>
#include <QMessageBox>
#include <QImage>
#include <QScreen>
#include <QGuiApplication>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    //ui设计的好 代码在1中

    ui->setupUi(this);
    setWindowTitle("自动寻址");

    runPath = QCoreApplication::applicationDirPath();       //获取exe路径
    hglpName = "photo";
    hglpPath = QString("%1/%2").arg(runPath).arg(hglpName);




    //图片框初始图片
    ui->label_img->setStyleSheet("QLabel{color:rgb(0,255,255);"
                               "border-image:url(:/new/prefix1/picture/p7.jpg)"

                             "}");

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::paintEvent(QPaintEvent *)
{
    QPainter Backp;  //添加背景图片
    Backp.begin(this);
    Backp.drawPixmap(0,0,width(),height(),QPixmap("../picture/1.png"));//背景这里需要优化

    Backp.end();
}


void MainWindow::on_ButtonStart_clicked()
{
    QMessageBox::about(this,"温馨提示","请输入图片的实际长度（单位为km）、图片想要分块的行数、列数"
                                   "  注:一共要输入三个值!");
}

void MainWindow::on_ButtonSelect_clicked()  //选择文件
{
    QString path = QFileDialog::getOpenFileName(this,"选择图片","../",
                                        "all(*.*)");
    qDebug()<<"图片路径"<<path;
    if(true == path.isEmpty()){
        return;
    }
    else{
        QImage img;
        if(!(img.load(path))){
            QMessageBox::information(this, tr("打开图像失败"),tr("打开图像失败!"));
                        return;
        }
        //ui->label_img->setPixmap(QPixmap::fromImage(img.scaled(ui->label_img->size())));

        ui->label_img->setPixmap(QPixmap(path));
        ui->label_img->setScaledContents(true);
        qDebug()<<"来啦"<<img.height();
        img_height = img.height();
        qDebug()<<"来啦0"<<img_height;
        qDebug()<<"来啦2"<<img.width();
    }

}

void MainWindow::on_ButtonSave_clicked()//保存文件
{
    QString filename1 = QFileDialog::getSaveFileName(this,tr("Save Image"),"",tr("Images (*.png *.bmp *.jpg)")); //选择路径
    QScreen *screen = QGuiApplication::primaryScreen();//产生异常，需要处理
    screen->grabWindow(ui->label_img->winId()).save(filename1);

}

void MainWindow::on_ButtonAbout_clicked()
{
    QMessageBox::about(this,"关于我们","我们是一个搞笑团队，由高校大学生组成");
}

void MainWindow::on_Buttonmethod_clicked()
{
    QMessageBox::about(this,"使用方法","首先选择保存在本地的一张图片，打开文件后点击开始即可");

}

void MainWindow::on_ButtonLong_clicked()
{
    qint32 longth = ui->lineEditLong->text().toInt();
    qDebug()<<"来过l"<<longth;

}

void MainWindow::on_ButtonRow_clicked()
{
    qint32 num = ui->lineEditRow->text().toInt();
    qDebug()<<"来过n"<<num;

}

void MainWindow::on_ButtonCol_clicked()
{
    qint32 num_c = ui->lineEditCol->text().toInt();
    qDebug()<<"来过n"<<num_c;
}

void MainWindow::on_ButtonPeople_clicked()
{
    qint32 People = ui->lineEditPeople->text().toInt();
    qDebug()<<"来过P"<<People;
}

void MainWindow::on_ButtonCar_clicked()
{
    qint32 car = ui->lineEditCar->text().toInt();
    qDebug()<<"来过car"<<car;
}
