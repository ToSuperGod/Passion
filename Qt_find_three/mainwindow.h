#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
protected:
    void paintEvent(QPaintEvent *);

private slots:
    void on_ButtonStart_clicked();

    void on_ButtonSelect_clicked();

    void on_ButtonSave_clicked();

    void on_ButtonAbout_clicked();

    void on_Buttonmethod_clicked();

    void on_ButtonLong_clicked();

    void on_ButtonRow_clicked();

    void on_ButtonCol_clicked();

    void on_ButtonPeople_clicked();

    void on_ButtonCar_clicked();

private:
    Ui::MainWindow *ui;
    QString runPath;       //获取exe路径
    QString hglpName;
    QString hglpPath;
    qint32 img_height;


};

#endif // MAINWINDOW_H
