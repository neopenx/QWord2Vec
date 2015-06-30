#include "widget.h"
#include "ui_widget.h"
#define doc pDocument
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);
    pDocument=new Document;
    pDocument->log=ui->plainTextEdit;
}

void Widget::updateStatus(QString msg)
{
    ui->plainTextEdit_2->setPlaceholderText(msg);
}

void Widget::TrainModel()
{
    long long a, b, c, d;
    FILE *fo;
    Thread **pt = new Thread*[doc->num_threads+1];
    qDebug("Starting training using file %s\n", doc->train_file);
    doc->vocab = (struct vocab_word *)calloc(doc->vocab_max_size, sizeof(struct vocab_word));
    doc->vocab_hash = (int *)calloc(doc->vocab_hash_size, sizeof(int));
    doc->expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (int i = 0; i < EXP_TABLE_SIZE; i++)
    {
        doc->expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        doc->expTable[i] = doc->expTable[i] / (doc->expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    doc->starting_alpha = doc->alpha;
    if (doc->read_vocab_file[0] != 0) doc->ReadVocab(); else doc->LearnVocabFromTrainFile();
    if (doc->save_vocab_file[0] != 0) doc->SaveVocab();
    if (doc->output_file[0] == 0) return;
    doc->InitNet();
    if (doc->negative > 0) doc->InitUnigramTable();
    for (a = 0; a < doc->num_threads; a++)
    {
        pt[a]=new Thread(a,pDocument);
        pt[a]->widget=this;
        pt[a]->start();
    }
    fo = fopen(doc->output_file, "wb");
    if (doc->classes == 0)
    {
        // Save the word vectors
        fprintf(fo, "%lld %lld\n", doc->vocab_size, doc->layer1_size);
        for (a = 0; a < doc->vocab_size; a++)
        {
          fprintf(fo, "%s ", doc->vocab[a].word);
          if (doc->binary)
              for (b = 0; b < doc->layer1_size; b++)
                  fwrite(&doc->syn0[a * doc->layer1_size + b], sizeof(real), 1, fo);
          else
              for (b = 0; b < doc->layer1_size; b++)
                  fprintf(fo, "%lf ", doc->syn0[a * doc->layer1_size + b]);
          fprintf(fo, "\n");
        }
    }
    else
    {
        // Run K-means on the word vectors
        int clcn = doc->classes, iter = 10, closeid;
        int *centcn = (int *)malloc(doc->classes * sizeof(int));
        int *cl = (int *)calloc(doc->vocab_size, sizeof(int));
        real closev, x;
        real *cent = (real *)calloc(doc->classes * doc->layer1_size, sizeof(real));
        for (a = 0; a < doc->vocab_size; a++) cl[a] = a % clcn;
        for (a = 0; a < iter; a++) {
          for (b = 0; b < clcn * doc->layer1_size; b++) cent[b] = 0;
          for (b = 0; b < clcn; b++) centcn[b] = 1;
          for (c = 0; c < doc->vocab_size; c++) {
            for (d = 0; d < doc->layer1_size; d++) cent[doc->layer1_size * cl[c] + d] += doc->syn0[c * doc->layer1_size + d];
            centcn[cl[c]]++;
          }
          for (b = 0; b < clcn; b++) {
            closev = 0;
            for (c = 0; c < doc->layer1_size; c++) {
              cent[doc->layer1_size * b + c] /= centcn[b];
              closev += cent[doc->layer1_size * b + c] * cent[doc->layer1_size * b + c];
            }
            closev = sqrt(closev);
            for (c = 0; c < doc->layer1_size; c++) cent[doc->layer1_size * b + c] /= closev;
          }
          for (c = 0; c < doc->vocab_size; c++) {
            closev = -10;
            closeid = 0;
            for (d = 0; d < clcn; d++) {
              x = 0;
              for (b = 0; b < doc->layer1_size; b++) x += cent[doc->layer1_size * d + b] * doc->syn0[c * doc->layer1_size + b];
              if (x > closev) {
                closev = x;
                closeid = d;
              }
            }
            cl[c] = closeid;
          }
        }
        // Save the K-means classes
        for (a = 0; a < doc->vocab_size; a++) fprintf(fo, "%s %d\n", doc->vocab[a].word, cl[a]);
        free(centcn);
        free(cent);
        free(cl);
     }
     fclose(fo);
}

Widget::~Widget()
{
    delete ui;
}

void Widget::on_pushButton_3_clicked()
{
    applySettings();
    TrainModel();
}

void Widget::applySettings()
{
    pDocument->alpha=ui->alpha->text().toFloat();
    pDocument->sample=ui->sample->text().toFloat();
    pDocument->window=ui->window->text().toInt();
    pDocument->layer1_size=ui->size->text().toInt();
    pDocument->classes=ui->classes->text().toInt();
    pDocument->negative=ui->negative->text().toInt();
    pDocument->iter=ui->iter->text().toInt();
    pDocument->num_threads=ui->thread->text().toInt();
    strcpy(pDocument->train_file,ui->input_file->text().toUtf8().constData());
    strcpy(pDocument->output_file,ui->output_file->text().toUtf8().constData());
    if(ui->cbow->isChecked()) pDocument->cbow=1;
    else pDocument->cbow=0;
    if(ui->hs->isChecked()) pDocument->hs=1;
    else pDocument->hs=0;
    if(ui->binary->isChecked()) pDocument->binary=1;
    else pDocument->binary=0;
}
