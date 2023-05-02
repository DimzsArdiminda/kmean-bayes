Kmeans
- sistem ini hanya menerima inputan dari dua kolom tabel, jika ingin menggunakan lebih dari satu kolom, 
  diperlukan beberapa konfigurasi
- dalam progam ini jumlah cluster yang ditentukan hanya 2, dengan konfigurasi sederhana, pengguna dapat
  mengubah jumlah peng-clusteran sesuai keinginan, tata cara mengubah jumlah cluster yaitu
	1. mengubah jumlah 'n_cluster' pada variabel km (line 33), variabel ini berfungsi menentukan jumlah cluster
	2. menambah jumlah variabel 'data#' disesuaikan dengan jumlah cluster yang ditentukan (mulai dari line 40), 
	   variabel ini berfungsi untuk memberi nama pada cluster (misal cluster pertama diberi nama 'satu') sesuai 
	   keinginan pengguna
	3. sesuaikan jumlah 'conditions' dan 'choices' pada jumlah cluster 

Naive Bayes
- pada progam naive bayes, data dibagi menjadi 2 bagian, yaitu testing set dan training set, dengan rasio 
  70% (training set) - 30% (testing set), rasio pembagian data bisa disesuaikan sesuai keinginan dengan melakukan
  beberapa konfigurasi pada database
- progam ini menggunakan 3 kolom inputan dengan 2 kolom untuk melakukan training dan testing, dan 1 kolom sebagai 
  target variables

