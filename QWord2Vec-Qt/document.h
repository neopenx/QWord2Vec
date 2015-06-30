#ifndef DOCUMENT_H
#define DOCUMENT_H
#include "cstdio"
#include "QStringList"
#include "QPlainTextEdit"
struct vocab_word
{
  long long cn;
  int *point;
  char *word, *code, codelen;
  bool operator < (const vocab_word &a) const {return cn>a.cn;} //Decreasing
};
typedef float real;
class Document
{
public:
    Document();
    void InitUnigramTable();
    void ReadWord(char *word, FILE *fin);
    int GetWordHash(char *word);
    int SearchVocab(char *word);
    int ReadWordIndex(FILE *fin);
    int AddWordToVocab(char *word);
    void LearnVocabFromTrainFile();
    void SortVocab();
    void ReduceVocab();
    void CreateBinaryTree();
    void SaveVocab();
    void ReadVocab();
    void InitNet();
    vocab_word *vocab;
    int *table,table_size=1e8,*vocab_hash,iter=5,num_threads = 12;
    int min_count=5,min_reduce=1,debug_mode=2;
    long long vocab_max_size=1000,vocab_size=0,layer1_size=100,vocab_hash_size=30000000,train_words=0;
    long long file_size,word_count_actual = 0;
    char train_file[100], output_file[100],save_vocab_file[100]={0}, read_vocab_file[100]={0};
    real alpha = 0.025, starting_alpha, sample = 1e-3;
    real *syn0, *syn1, *syn1neg, *expTable;
    int hs = 0, negative = 5,cbow = 1,window = 5,classes=0,binary=1;
    QPlainTextEdit *log;

};

#endif // DOCUMENT_H
