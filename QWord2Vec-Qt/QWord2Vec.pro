#-------------------------------------------------
#
# Project created by QtCreator 2015-06-28T22:55:15
#
#-------------------------------------------------

QT       += core gui
CONFIG += c++11
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = QWord2Vec
TEMPLATE = app


SOURCES += main.cpp\
        widget.cpp \
    document.cpp \
    thread.cpp

HEADERS  += widget.h \
    document.h \
    thread.h

FORMS    += widget.ui
