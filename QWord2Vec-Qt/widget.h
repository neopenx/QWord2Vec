#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include "thread.h"
class Thread;
namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = 0);
    void TrainModel();
    Document *pDocument;
    void applySettings();
    Thread **pt;
    void updateStatus(QString msg);
    ~Widget();

private slots:
    void on_pushButton_3_clicked();

private:
    Ui::Widget *ui;
};

#endif // WIDGET_H
