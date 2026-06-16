---
title: Segmentacja semantyczna - AeroScapes
authors:
  - Kacper Chmielewski
  - Patryk Maciąg
  - Adrian Walczak
  - Weronika Tarnawska
options:
  implicit_slide_ends: true
  end_slide_shorthand: true
  command_prefix: "cmd: "
  image_attributes_prefix: ""
theme:
  override:
  max_rows_alignment: center
---

Segmentacja semantyczna w AeroScapes
===

- obraz z drona -> klasa dla każdego piksela
- 11 klas + tło
- trudne: małe obiekty, zmienna skala, nierówne klasy

Cel projektu
===

- wejście: obraz RGB z drona
- wyjście: mapa klas pikselowych
- funkcja: `f: R^(H×W×3) -> C^(H×W)`
- cel: poprawna klasyfikacja każdego piksela
- problem: rzadkie klasy łatwo giną

Dataset
===

- AeroScapes / UAV segmentation
- 3269 par obraz-mask
- obrazy: 1280×720 RGB
- maski: jeden kanał, wartości `0..11`
- split: oficjalne `train / val`

EDA - jak wyglądają dane
===

- wszystkie obrazy mają ten sam rozmiar
- maski są indeksami klas, bez dekodowania kolorów
- normalizacja: ImageNet mean/std
<!-- - resize: `BILINEAR` dla obrazu, `NEAREST` dla maski -->

EDA - nierównowaga klas
===

- dominują: vegetation, road, background
- person / bike / obstacle: małe, ale częste
- boat / animal / drone: naprawdę rzadkie
- sam loss nie wystarczy -> potrzebne wagi klas

![width:100%](./tmp/6_eda_clacc_imbalance.png)

Wnioski z EDA
===

- używamy wag klas wyznaczonych metodą median frequency balancing (klasy o mniejszym udziale pikseli otrzymują większą wagę)
- trenujemy na cropach 512×512
- unikamy agresywnego zmniejszania obrazów, ponieważ może to spowodować zniknięcie małych obiektów
- background ignorujemy w lossie

<!-- ![width:100%](./tmp/table3_class_weights.png) -->

Porównanie z rozwiązaniami referencyjnymi
===

![width:100%](./tmp/table1.png)

- najlepsze publiczne wyniki osiągają ok. **0.93 mIoU**
<!-- - porównanie z naszymi wynikami nie jest bezpośrednie:
  - wiele prac używa **rozdzielczości 256×256**, a my walidujemy na **1280×720**
  - często model przewiduje tylko **11 klas** (bez tła), podczas gdy u nas tło jest obecne i ignorowane jedynie w funkcji straty oraz metryce (`ignore_index=0`)
  - stosowane są inne backbone'y, sposoby pretrenowania i pipeline'y treningowe -->

Baseline
===

- DeepLabV3-ResNet50
- backbone zamrożony
- dotrenowana tylko ostatnia warstwa
- loss: CrossEntropy z wagami klas z EDA
<!-- - punkt odniesienia do dalszych eksperymentów -->

Metoda A: Dice + Weighted Cross Entropy
===

**Cel:** poprawa wyników dla rzadkich i małych klas

- architektura: **DeepLabV3 + ResNet50**
- backbone z pretrenowaniem **ImageNet**
- trening na losowych cropach **512×512**
- funkcja straty:
  - **Weighted Cross Entropy**
  - **Dice Loss**
- wagi klas wyznaczone na podstawie EDA (median-frequency)
- walidacja na pełnej rozdzielczości **1280×720**

<!-- **Efekt:** największy wzrost jakości spośród testowanych metod -->
<!-- **+0.037 mIoU względem baseline** -->

Metoda B: mocniejszy backbone
===

**Cel:** zwiększenie maksymalnego mIoU

- architektura: **DeepLabV3 + ResNet101**
- pełne pretrenowanie na **COCO**
- ten sam loss co w metodzie A (**Dice + Weighted CE**)
- trening na cropach **512×512**
- walidacja na **1280×720**

<!-- **Efekt:** poprawa względem baseline o **+0.022 mIoU**, ale mniejsza niż przy samej zmianie funkcji straty. -->

Nasze metody
===

- wspólny pipeline treningu
- wariant A: Dice + weighted CE
- wariant B: ResNet101 + COCO + Dice + CE
- ta sama walidacja: 1280×720

![width:100%](./tmp/table2.png)

Wyniki zbiorcze
===

- baseline CE: `0.672`
- metoda A: `0.709`
- metoda B: `0.694`
- najlepszy zysk daje zmiana lossu, nie większy backbone

Wyniki - mIoU w kolejnych epokach
===

![width:100%](./tmp/1_val_miou_per_epoch.png)

Wyniki - validation loss
===

![width:100%](./tmp/2_val_loss_noise.png)

Wyniki - IoU per klasa
===

![width:100%](./tmp/3_per_class_iou.png)

Trudne klasy
===

- bike, obstacle i animal poprawiają się, ale zostają poniżej łatwych klas
- metoda A i B wygrywają na różnych klasach
<!-- - małe obiekty cierpią przez błędy na granicach -->

![width:100%](./tmp/4_A_hard_easy_classes.png)

Analiza jakościowa
===

![width:100%](./tmp/5_quantitative_analysis.png)

Analiza jakościowa
===

- **Małe, cienkie obiekty.** Rowery i przeszkody zajmują kilka pikseli, więc nawet minimalne przesunięcie granicy mocno obniża IoU, mimo że maska wygląda sensownie.
- **Niejasna klasa *przeszkoda***. Obejmuje słupki, kamienie, śmieci, ogrodzenia — mało spójnych cech, więc niskie IoU wynika też z niejednoznacznych etykiet.
- **Różne skale.** Zmieniająca się wysokość kamery sprawia, że ten sam obiekt ma różną wielkość, a małe instancje łatwo giną.

Dalsze kierunki rozwoju
===

- trenowanie na większych cropach lub wyższej rozdzielczości
- dalsze eksperymenty z funkcją straty (większy udział Dice, Focal Loss)
- copy-paste augmentation dla rzadkich klas (bike, obstacle, animal)
- połączenie dwóch wyspecjalizowanych modeli wykorzystujących mocne strony metod A i B

Podsumowanie
===

- **Dice + Weighted Cross Entropy** okazało się skuteczniejsze niż samo zwiększenie pojemności modelu

- największym wyzwaniem pozostają małe i rzadkie obiekty (bike, obstacle, animal)

- metody A i B osiągają najlepsze wyniki dla różnych klas, co sugeruje potencjał ich połączenia

- końcowy model poprawił wynik baseline'u o **3.7 punktu procentowego mIoU** przy zachowaniu tej samej architektury bazowej


Repo
===

https://github.com/WeronikaTarnawska/projekt_nn-aeroscapes_semantic_segmentation
