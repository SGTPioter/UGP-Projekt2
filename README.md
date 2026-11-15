# Klasyfikacja obrazów CIFAR-10 z użyciem CNN i MobileNetV2

Projekt porównuje własną sieć konwolucyjną trenowaną od podstaw z MobileNetV2 wykorzystującym wagi z ImageNet. Wpływ augmentacji sprawdzono w osobnym przebiegu własnej CNN. **W tym eksperymencie dostrojony MobileNetV2 osiągnął najwyższą trafność testową (88,76%)**, o 0,80 pkt proc. więcej niż CNN z augmentacją.

[Raport PDF](Proj2Rap.pdf) · [Wyniki CSV](results/20261113-005142-732453/comparison.csv) · [Ustawienia przebiegu](results/20261113-005142-732453/settings.json)

## Dane i warianty

CIFAR-10 zawiera 60 000 kolorowych obrazów 32 × 32 w 10 klasach. Z oficjalnej części treningowej wydzielono w każdej klasie po 4 500 obrazów do treningu i 500 do walidacji. Wszystkie warianty korzystały z tego samego podziału obejmującego 45 000 obrazów treningowych, 5 000 walidacyjnych i pełny oficjalny test z 10 000 obrazów.

![Przykładowe obrazy z dziesięciu klas CIFAR-10](results/20261113-005142-732453/class_examples.png)

| Wariant | Zakres treningu |
| --- | --- |
| Własna CNN | Trzy bloki konwolucyjne i klasyfikator uczone od podstaw. |
| CNN z augmentacją | Ta sama architektura i podział danych. Podczas treningu stosowano losowe odbicia poziome i przesunięcia do 10% rozmiaru obrazu. |
| MobileNetV2 z zamrożoną bazą | Baza obliczała 1 280 cech dla obrazu powiększonego do 96 × 96. Uczono tylko głowicę z 10 wyjściami. |
| MobileNetV2 po fine-tuningu | Kontynuowano trening głowicy i końcowych warstw bazy od `block_13_expand`, zachowując zamrożone warstwy Batch Normalization. |

## Trening i wybór wag

Użyto optymalizatora AdamW z `weight decay` równym 0,0001, z wyłączeniem biasów oraz parametrów skali i przesunięcia warstw Batch Normalization. Dane przetwarzano w partiach po maksymalnie 128 przykładów. Początkowy współczynnik uczenia wynosił 0,001 dla obu CNN i głowicy oraz 0,0001 dla fine-tuningu.

Limit wynosił 1 000 epok na etap. Po 4 epokach bez poprawy straty walidacyjnej współczynnik uczenia zmniejszano o połowę. Early stopping kończył etap po 15 epokach bez poprawy i przywracał wagi z najniższą stratą walidacyjną.

| Wariant | Wykonane epoki | Epoka wybranych wag | Strata walidacyjna | Trafność walidacyjna |
| --- | ---: | ---: | ---: | ---: |
| Własna CNN | 53 | 38 | 0,4280 | 87,12% |
| CNN z augmentacją | 88 | 73 | 0,3399 | 88,60% |
| MobileNetV2 z zamrożoną bazą | 22 | 7 | 0,3941 | 86,84% |
| MobileNetV2 po fine-tuningu | 17 | 2 | 0,3427 | 88,94% |

W fine-tuningu najwyższa trafność walidacyjna wyniosła 91,86% w 17. epoce. Do testu użyto wag z 2. epoki, w której strata walidacyjna była najniższa.

## Wyniki testowe i koszt obliczeniowy

| Wariant | Trafność | Precyzja macro | F1 macro | Strata testowa |
| --- | ---: | ---: | ---: | ---: |
| Własna CNN | 86,16% | 0,8616 | 0,8608 | 0,4886 |
| CNN z augmentacją | 87,96% | 0,8790 | 0,8788 | 0,3643 |
| MobileNetV2 z zamrożoną bazą | 86,26% | 0,8633 | 0,8625 | 0,4009 |
| MobileNetV2 po fine-tuningu | **88,76%** | **0,8907** | **0,8877** | **0,3436** |

Augmentacja poprawiła trafność własnej CNN o **1,80 pkt proc.** Własna CNN bez augmentacji i zamrożona baza uzyskały zbliżoną trafność, a różnica odpowiadała 10 poprawnym klasyfikacjom na 10 000 obrazów.

| Wariant | Parametry trenowane | Parametry ogółem | Łączny trening | Wnioskowanie [ms/obraz] |
| --- | ---: | ---: | ---: | ---: |
| Własna CNN | 288 746 | 289 642 | 58 min 18 s | 0,192 |
| CNN z augmentacją | 288 746 | 289 642 | 99 min 13 s | 0,195 |
| MobileNetV2 z zamrożoną bazą | 12 810 | 2 270 794 | 2 min 20 s | 2,522 |
| MobileNetV2 po fine-tuningu | 1 676 170 | 2 270 794 | 43 min 32 s | 2,612 |

Czasy mierzono na procesorze Intel Core i7-10700KF. Czas treningu MobileNetV2 uwzględnia ekstrakcję cech i trening głowicy, a w wariancie dostrojonym także dodatkowy etap uczenia. Czas wnioskowania w przeliczeniu na obraz to mediana 15 pomiarów dla partii 128 obrazów po trzech pomiarach rozgrzewkowych, podzielona przez 128.

- **Własna CNN z augmentacją** sprawdziła się, gdy liczył się czas wnioskowania. Przetwarzała partię około 13 razy szybciej niż dostrojony MobileNetV2 na badanym CPU.
- **Zamrożona baza** pozwoliła przygotować model w najkrótszym czasie. Ekstrakcja cech i trening głowicy zajęły łącznie 2 min 20 s.
- **Fine-tuning** poprawił trafność o 2,50 pkt proc. względem zamrożonej bazy przy zbliżonym czasie wnioskowania.

Każdy wariant wytrenowano raz, więc wnioski dotyczą tego przebiegu. Wpływ augmentacji oceniono dla własnej CNN. Przy partii z jednym obrazem lub na GPU proporcje czasów wnioskowania mogą być inne.

## Wykresy i macierze pomyłek

W macierzach wiersz odpowiada klasie rzeczywistej, a kolumna klasie przewidzianej. Kolor oznacza odsetek w obrębie rzeczywistej klasy. Po augmentacji odsetek kotów sklasyfikowanych jako psy spadł z 13,2% do 9,1%, a odsetek psów sklasyfikowanych jako koty wzrósł z 8,2% do 9,0%. Fine-tuning zwiększył czułość dla ptaków z 79,7% do 91,0%, a dla koni czułość spadła z 87,1% do 83,2%.

| Wariant | Krzywe uczenia | Macierz pomyłek |
| --- | --- | --- |
| Własna CNN | [Wykres](results/20261113-005142-732453/cnn/learning_curves.png) | [Macierz](results/20261113-005142-732453/cnn/confusion_matrix.png) |
| CNN z augmentacją | [Wykres](results/20261113-005142-732453/cnn_augmented/learning_curves.png) | [Macierz](results/20261113-005142-732453/cnn_augmented/confusion_matrix.png) |
| MobileNetV2 z zamrożoną bazą | [Wykres](results/20261113-005142-732453/feature_extraction/learning_curves.png) | [Macierz](results/20261113-005142-732453/feature_extraction/confusion_matrix.png) |
| MobileNetV2 po fine-tuningu | [Wykres](results/20261113-005142-732453/fine_tuning/learning_curves.png) | [Macierz](results/20261113-005142-732453/fine_tuning/confusion_matrix.png) |

## Uruchomienie

W terminalu z aktywnym środowiskiem Python 3.11, w katalogu projektu, uruchom:

```sh
python -m pip install "tensorflow==2.21.0" "keras==3.12.4" numpy scikit-learn matplotlib
python main.py --train-per-class 4500 --val-per-class 500 --test-per-class 1000 --epochs 1000 --fine-tune-epochs 1000 --batch-size 128 --seed 2026
```

Keras pobiera CIFAR-10 i wagi ImageNet przy pierwszym użyciu. Każdy przebieg tworzy osobny podfolder w `results/`, gdzie zapisuje ustawienia, metryki, wykresy i modele.
