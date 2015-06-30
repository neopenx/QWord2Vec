#include "thread.h"
#include "QDebug"
#define MAX_SENTENCE_LENGTH 1000
#define MAX_EXP 6
#define EXP_TABLE_SIZE 1000
#define doc pDocument
Thread::Thread(long long id,Document *pDocument):id(id),pDocument(pDocument) {}

void Thread::run()
{
    qDebug()<<currentThreadId();
    //widget->updateStatus("sss");
    long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    //qDebug("233  %d\n",doc->iter);
    long long l1, l2, c, target, label, local_iter = doc->iter;
    unsigned long long next_random =id;
    real f, g;
    //clock_t now;
    real *neu1 = (real *)calloc(doc->layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(doc->layer1_size, sizeof(real));
    FILE *fi = fopen(doc->train_file, "rb");
    fseek(fi, doc->file_size / (long long)doc->num_threads * (long long)id, SEEK_SET);
    while (1)
    {
        if (word_count - last_word_count > 10000)
        {
          doc->word_count_actual += word_count - last_word_count;
          last_word_count = word_count;
          if ((doc->debug_mode > 1))
          {
            //now=clock();
            widget->updateStatus(QString("Alpha: %1  Progress: %2%%  ").arg(QString::number(doc->alpha))
                                 .arg(QString::number(doc->word_count_actual / (real)(doc->iter * doc->train_words + 1) * 100)));
            /*log->setPlainText(QString("Alpha: %1  Progress: %2%%  ").arg(QString::number(doc->alpha))
                              .arg(QString::number(doc->word_count_actual / (real)(doc->iter * doc->train_words + 1) * 100)));*/
            /*qDebug("%cAlpha: %f  Progress: %.2f%%  ", 13, doc->alpha,
             doc->word_count_actual / (real)(doc->iter * doc->train_words + 1) * 100);*/
            fflush(stdout);
              //qDebug("2333\n");
          }
          doc->alpha = doc->starting_alpha * (1 - doc->word_count_actual / (real)(doc->iter * doc->train_words + 1));
          if (doc->alpha < doc->starting_alpha * 0.0001)
                  doc->alpha = doc->starting_alpha * 0.0001;
        }
        if (sentence_length == 0)
        {
          while (1)
          {
            word = doc->ReadWordIndex(fi);
            if (feof(fi)) break;
            if (word == -1) continue;
            word_count++;
            if (word == 0) break;
            // The subsampling randomly discards frequent words while keeping the ranking same
            if (doc->sample > 0) {
              real ran = (sqrt(doc->vocab[word].cn / (doc->sample * doc->train_words)) + 1) * (doc->sample * doc->train_words) / doc->vocab[word].cn;
              next_random = next_random * (unsigned long long)25214903917 + 11;
              if (ran < (next_random & 0xFFFF) / (real)65536) continue;
            }
            sen[sentence_length] = word;
            sentence_length++;
            if (sentence_length >= MAX_SENTENCE_LENGTH) break;
          }
          sentence_position = 0;
        }
        if (feof(fi) || (word_count > doc->train_words / doc->num_threads))
        {
          doc->word_count_actual += word_count - last_word_count;
          local_iter--;
          if (local_iter == 0) break;
          word_count = 0;
          last_word_count = 0;
          sentence_length = 0;
          fseek(fi, doc->file_size / (long long)doc->num_threads * (long long)id, SEEK_SET);
          continue;
        }
        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < doc->layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < doc->layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % doc->window;
        if (doc->cbow)
        {  //train the cbow architecture
          // in -> hidden
          cw = 0;
          for (a = b; a < doc->window * 2 + 1 - b; a++) if (a != doc->window)
          {
            c = sentence_position - doc->window + a;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;
            for (c = 0; c < doc->layer1_size; c++) neu1[c] += doc->syn0[c + last_word * doc->layer1_size];
            cw++;
          }
          if (cw)
          {
            for (c = 0; c < doc->layer1_size; c++) neu1[c] /= cw;
            if (doc->hs) for (d = 0; d < doc->vocab[word].codelen; d++)
            {
              f = 0;
              l2 = doc->vocab[word].point[d] * doc->layer1_size;
              // Propagate hidden -> output
              for (c = 0; c < doc->layer1_size; c++) f += neu1[c] * doc->syn1[c + l2];
              if (f <= -MAX_EXP) continue;
              else if (f >= MAX_EXP) continue;
              else f = doc->expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
              // 'g' is the gradient multiplied by the learning rate
              g = (1 - doc->vocab[word].code[d] - f) * doc->alpha;
              // Propagate errors output -> hidden
              for (c = 0; c < doc->layer1_size; c++) neu1e[c] += g * doc->syn1[c + l2];
              // Learn weights hidden -> output
              for (c = 0; c < doc->layer1_size; c++) doc->syn1[c + l2] += g * neu1[c];
            }
            // NEGATIVE SAMPLING
            if (doc->negative > 0)
                for (d = 0; d < doc->negative + 1; d++)
                {
                    if (d == 0)
                    {
                        target = word;
                        label = 1;
                    }
                    else
                    {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        target = doc->table[(next_random >> 16) % doc->table_size];
                        if (target == 0) target = next_random % (doc->vocab_size - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }
                    l2 = target * doc->layer1_size;
                    f = 0;
                    for (c = 0; c < doc->layer1_size; c++) f += neu1[c] * doc->syn1neg[c + l2];
                    if (f > MAX_EXP) g = (label - 1) * doc->alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * doc->alpha;
                    else g = (label - doc->expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * doc->alpha;
                    for (c = 0; c < doc->layer1_size; c++) neu1e[c] += g * doc->syn1neg[c + l2];
                    for (c = 0; c < doc->layer1_size; c++) doc->syn1neg[c + l2] += g * neu1[c];
                }
            // hidden -> in
            for (a = b; a < doc->window * 2 + 1 - b; a++)
                if (a != doc->window)
                {
                    c = sentence_position - doc->window + a;
                    if (c < 0) continue;
                    if (c >= sentence_length) continue;
                    last_word = sen[c];
                    if (last_word == -1) continue;
                    for (c = 0; c < doc->layer1_size; c++) doc->syn0[c + last_word * doc->layer1_size] += neu1e[c];
                }
          }
        }
        else
        {  //train skip-gram
            for (a = b; a < doc->window * 2 + 1 - b; a++)
                if (a != doc->window)
                {
                    c = sentence_position - doc->window + a;
                    if (c < 0) continue;
                    if (c >= sentence_length) continue;
                    last_word = sen[c];
                    if (last_word == -1) continue;
                    l1 = last_word * doc->layer1_size;
                    for (c = 0; c < doc->layer1_size; c++) neu1e[c] = 0;
                    // HIERARCHICAL SOFTMAX
                    if (doc->hs)
                        for (d = 0; d < doc->vocab[word].codelen; d++)
                        {
                            f = 0;
                            l2 = doc->vocab[word].point[d] * doc->layer1_size;
                            // Propagate hidden -> output
                            for (c = 0; c < doc->layer1_size; c++) f += doc->syn0[c + l1] * doc->syn1[c + l2];
                            if (f <= -MAX_EXP) continue;
                            else if (f >= MAX_EXP) continue;
                            else f = doc->expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                            // 'g' is the gradient multiplied by the learning rate
                            g = (1 - doc->vocab[word].code[d] - f) * doc->alpha;
                            // Propagate errors output -> hidden
                            for (c = 0; c < doc->layer1_size; c++) neu1e[c] += g * doc->syn1[c + l2];
                            // Learn weights hidden -> output
                            for (c = 0; c < doc->layer1_size; c++) doc->syn1[c + l2] += g * doc->syn0[c + l1];
                        }
            // NEGATIVE SAMPLING
            if (doc->negative > 0)
                for (d = 0; d < doc->negative + 1; d++)
                {
                    if (d == 0)
                    {
                        target = word;
                        label = 1;
                    }
                    else
                    {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        target = doc->table[(next_random >> 16) % doc->table_size];
                        if (target == 0) target = next_random % (doc->vocab_size - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }
                    l2 = target * doc->layer1_size;
                    f = 0;
                    for (c = 0; c < doc->layer1_size; c++) f += doc->syn0[c + l1] * doc->syn1neg[c + l2];
                    if (f > MAX_EXP) g = (label - 1) * doc->alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * doc->alpha;
                    else g = (label - doc->expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * doc->alpha;
                    for (c = 0; c < doc->layer1_size; c++) neu1e[c] += g * doc->syn1neg[c + l2];
                    for (c = 0; c < doc->layer1_size; c++) doc->syn1neg[c + l2] += g * doc->syn0[c + l1];
                }
            // Learn weights input -> hidden
            for (c = 0; c < doc->layer1_size; c++) doc->syn0[c + l1] += neu1e[c];
          }
        }
        sentence_position++;
        if (sentence_position >= sentence_length)
        {
          sentence_length = 0;
          continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    quit();
    //QThread::run();
}

Thread::~Thread()
{

}

