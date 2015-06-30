#include "document.h"
#include "cstdlib"
#include "cmath"
#include "cstring"
#include "algorithm"
#include "QDebug"
#include "QFile"
using namespace std;
#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
Document::Document()
{

}

void Document::InitUnigramTable()
{
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

void Document::ReadWord(char *word, FILE *fin)
{
  int a = 0, ch;
  while (!feof(fin))
  {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
    {
      if (a > 0)
      {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n')
      {
        strcpy(word, (char *)"</s>");
        return;
      }
      else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

int Document::GetWordHash(char *word)
{
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

int Document::SearchVocab(char *word)
{
  unsigned int hash = GetWordHash(word);
  while (1)
  {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

int Document::ReadWordIndex(FILE *fin)
{
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

int Document::AddWordToVocab(char *word)
{
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size)
  {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

void Document::SortVocab()
{
  int a, size;unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  sort(vocab+1,vocab+vocab_size);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++)
  {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0))
    {
      vocab_size--;
      free(vocab[a].word);
    }
    else
    {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++)
  {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

void Document::ReduceVocab()
{
  int a, b = 0;unsigned int hash;
  for (a = 0; a < vocab_size; a++)
      if (vocab[a].cn > min_reduce)
      {
          vocab[b].cn = vocab[a].cn;
          vocab[b].word = vocab[a].word;
          b++;
      }
      else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++)
  {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void Document::CreateBinaryTree()
{
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++)
  {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0)
    {
      if (count[pos1] < count[pos2]) {min1i = pos1;pos1--;}
      else {min1i = pos2;pos2++;}
    }
    else {min1i = pos2;pos2++;}
    if (pos1 >= 0)
    {
      if (count[pos1] < count[pos2]) {min2i = pos1;pos1--;}
      else {min2i = pos2;pos2++;}
    }
    else {min2i = pos2;pos2++;}
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++)
  {
    b = a;i = 0;
    while (1)
    {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++)
    {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void Document::LearnVocabFromTrainFile()
{
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL)
  {
    printf("ERROR: training data file not found!\n");
    log->appendPlainText(QString("ERROR: training data file not found!\n"));
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  log->appendPlainText(QString("Now Reading Words.....\n"));
  while (1)
  {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 1000000 == 0))
    {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1)
    {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0)
  {
    qDebug("Vocab size: %lld\n", vocab_size);
    qDebug("Words in train file: %lld\n", train_words);
    log->appendPlainText(QString("Vocab size: %1\n").arg(vocab_size));
    log->appendPlainText(QString("Words in train file: %1\n").arg(train_words));
  }
  file_size = ftell(fin);
  fclose(fin);
}

void Document::SaveVocab()
{
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void Document::ReadVocab()
{
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL)
  {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1)
  {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0)
  {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL)
  {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void Document::InitNet()
{
  long long a, b;
  unsigned long long next_random = 1;
  syn0=new real[(long long)vocab_size * layer1_size];
  //syn0=(real *)_aligned_malloc(64,(long long)vocab_size * layer1_size * sizeof(real));
  //a = _aligned_malloc((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs)
  {
    //a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    //syn1=(real *)_aligned_malloc(128,(long long)vocab_size * layer1_size * sizeof(real));
    syn1=new real[(long long)vocab_size * layer1_size];
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++)
        for (b = 0; b < layer1_size; b++)
           syn1[a * layer1_size + b] = 0;
  }
  if (negative>0)
  {
    //a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    //syn1neg=(real *)_aligned_malloc(128,(long long)vocab_size * layer1_size * sizeof(real));
    syn1neg=new real[(long long)vocab_size * layer1_size];
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++)
        for (b = 0; b < layer1_size; b++)
          syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
      {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
      }
  CreateBinaryTree();
}

