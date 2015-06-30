#ifndef THREAD_H
#define THREAD_H

#include "QThread"
#include "document.h"
#include "widget.h"
class Widget;
class ThreadExecuter:public QObject
{
    Q_OBJECT
public:
    ThreadExecuter();
};

class Thread : public QThread
{
    Q_OBJECT
public:
    Thread(long long id, Document *pDocument);
    Document *pDocument;
    Widget *widget;
    long long id;
    void setDocument(Document *doc) {pDocument=doc;}
    ~Thread();
protected:
    void run();
};

#endif // THREAD_H
