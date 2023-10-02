import torch
import os


class Vocab:
    def __init__(self, fichier_matrice: str) -> None:
        self.dico_voca = {}
        with open(fichier_matrice, 'r', encoding='utf8') as fi:
            ligne = fi.readline()
            ligne = ligne.strip()
            
            #self.emb_dim, self.vocab_size = eval(ligne)
            self.vocab_size, self.emb_dim = map(int,ligne.split(" "))
            self.matrice = torch.zeros((self.vocab_size, self.emb_dim))
            indice_mot = 0
        
            ligne = fi.readline()
            ligne = ligne.strip()
            while ligne != '': 
            
                splitted_ligne = ligne.split()
                self.dico_voca[splitted_ligne[0]] = indice_mot
                for i in range(1,len(splitted_ligne)):
                    self.matrice[indice_mot, i-1] = float(splitted_ligne[i])
                indice_mot += 1
                ligne = fi.readline()
                ligne = ligne.strip()

    def get_word_index(self, mot: str) -> int:
        if not mot in self.dico_voca:
            return None
        return self.dico_voca[mot]
                
    def get_emb(self, mot: str) -> torch.Tensor:
        if not mot in self.dico_voca:
            return None
        return  self.matrice[self.dico_voca[mot]]
    
    def get_emb_torch(self, indice_mot: int) -> torch.Tensor:
        # OPTIMISATION: no verificaiton allows to get embeddings a bit faster
        #if indice_mot < 0 or indice_mot >= self.matrice.shape()[0]: # not valid index
        #    return None
        #return self.matrice[indice_mot]
        return self.matrice[indice_mot]
        
    def get_one_hot(self, mot: str) -> torch.Tensor:
        vect = torch.zeros(len(self.dico_voca))
        vect[self.dico_voca[mot]] = 1
        return vect


if __name__ == '__main__':
    path = os.path.join('data', 'embeddings-word2vecofficial.train.unk5.txt')
    voc = Vocab(path)
    print(voc.get_emb('le'))
    idx = voc.get_word_index('le')
    print(voc.get_emb_torch(idx))
    print(voc.dico_voca)