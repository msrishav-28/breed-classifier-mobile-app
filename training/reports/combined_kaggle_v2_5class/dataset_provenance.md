# Dataset Preparation: Kaggle Indian Bovine Breeds v5 + Indian Buffalo Dataset v1 + Breed Cattle Buffalo v1

- Source: https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds; https://www.kaggle.com/datasets/atharvadarpude/indian-buffalo-dataset; https://www.kaggle.com/datasets/algsoch/breed-cattle-buffalo
- License: CC0: Public Domain; CC0: Public Domain; MIT
- Source directories: `C:\Users\Vignesh\.cache\kagglehub\datasets\lukex9442\indian-bovine-breeds\versions\5\Indian_bovine_breeds`, `C:\Users\Vignesh\.cache\kagglehub\datasets\atharvadarpude\indian-buffalo-dataset\versions\1\buffalo`, `C:\Users\Vignesh\.cache\kagglehub\datasets\algsoch\breed-cattle-buffalo\versions\1\clean_images`
- Output directory: `datasets\processed\combined_kaggle_v2_5class`
- Cleaning: corrupt/tiny images dropped; dHash near-duplicates dropped with distance <= 0.
- Hold-out test split: 30 images per included class.

## Included Classes

| Label | Raw | Cleaned | Train/Val | Test | Decision |
|---|---:|---:|---:|---:|---|
| Gir | 1140 | 396 | 366 | 30 | included |
| Hallikar | 488 | 191 | 161 | 30 | included below target 300; report as limited data |
| Murrah | 682 | 208 | 178 | 30 | included below target 300; report as limited data |
| Sahiwal | 894 | 374 | 344 | 30 | included |
| Tharparkar | 447 | 180 | 150 | 30 | included below target 300; report as limited data |

## Excluded Classes

| Raw class | Resolved label | Raw | Cleaned | Decision |
|---|---|---:|---:|---|
| buffalo/banni |  | 127 | 0 | excluded: no catalog label match |
| buffalo/bargur |  | 122 | 0 | excluded: no catalog label match |
| buffalo/bhadwari | Bhadawari | 148 | 0 | excluded: not in explicit include list |
| buffalo/Chhattisgarhi |  | 131 | 0 | excluded: no catalog label match |
| buffalo/chilika |  | 144 | 0 | excluded: no catalog label match |
| buffalo/gojri |  | 177 | 0 | excluded: no catalog label match |
| buffalo/Jaffarabadi | Jaffarabadi | 357 | 0 | excluded: not in explicit include list |
| buffalo/kalahandi |  | 67 | 0 | excluded: no catalog label match |
| buffalo/luit |  | 129 | 0 | excluded: no catalog label match |
| buffalo/marathwada |  | 172 | 0 | excluded: no catalog label match |
| buffalo/mehsana | Mehsana | 68 | 0 | excluded: not in explicit include list |
| buffalo/nagpuri | Nagpuri | 156 | 0 | excluded: not in explicit include list |
| buffalo/nili-ravi | Nili-Ravi | 189 | 0 | excluded: not in explicit include list |
| buffalo/pandharpuri | Pandharpuri | 141 | 0 | excluded: not in explicit include list |
| buffalo/surti | Surti | 170 | 0 | excluded: not in explicit include list |
| buffalo/toda |  | 161 | 0 | excluded: no catalog label match |
| clean_images/alambadi |  | 207 | 0 | excluded: no catalog label match |
| clean_images/amritmahal | Amritmahal | 204 | 0 | excluded: not in explicit include list |
| clean_images/ayrshire |  | 726 | 0 | excluded: no catalog label match |
| clean_images/bachaur |  | 10 | 0 | excluded: no catalog label match |
| clean_images/badri |  | 10 | 0 | excluded: no catalog label match |
| clean_images/banni |  | 119 | 0 | excluded: no catalog label match |
| clean_images/bargur |  | 122 | 0 | excluded: no catalog label match |
| clean_images/bargur |  | 102 | 0 | excluded: no catalog label match |
| clean_images/belahi |  | 14 | 0 | excluded: no catalog label match |
| clean_images/bhadawari | Bhadawari | 196 | 0 | excluded: not in explicit include list |
| clean_images/binjharpuri |  | 10 | 0 | excluded: no catalog label match |
| clean_images/brown_swiss |  | 469 | 0 | excluded: no catalog label match |
| clean_images/chhattisgarhi |  | 10 | 0 | excluded: no catalog label match |
| clean_images/chilika |  | 10 | 0 | excluded: no catalog label match |
| clean_images/dagri |  | 10 | 0 | excluded: no catalog label match |
| clean_images/dangi |  | 100 | 0 | excluded: no catalog label match |
| clean_images/deoni | Deoni | 222 | 0 | excluded: not in explicit include list |
| clean_images/gangatiri |  | 14 | 0 | excluded: no catalog label match |
| clean_images/gaolao |  | 14 | 0 | excluded: no catalog label match |
| clean_images/ghumusari |  | 10 | 0 | excluded: no catalog label match |
| clean_images/gojri |  | 14 | 0 | excluded: no catalog label match |
| clean_images/guernsey |  | 244 | 0 | excluded: no catalog label match |
| clean_images/hariana | Hariana | 140 | 0 | excluded: not in explicit include list |
| clean_images/himachali_pahari |  | 8 | 0 | excluded: no catalog label match |
| clean_images/holstein-friesian |  | 566 | 0 | excluded: no catalog label match |
| clean_images/jaffarabadi | Jaffarabadi | 112 | 0 | excluded: not in explicit include list |
| clean_images/jersey |  | 203 | 0 | excluded: no catalog label match |
| clean_images/kalahandi |  | 10 | 0 | excluded: no catalog label match |
| clean_images/kangayam | Kangayam | 101 | 0 | excluded: not in explicit include list |
| clean_images/kankrej |  | 193 | 0 | excluded: no catalog label match |
| clean_images/kenkatha |  | 69 | 0 | excluded: no catalog label match |
| clean_images/khariar |  | 10 | 0 | excluded: no catalog label match |
| clean_images/kherigarh |  | 46 | 0 | excluded: no catalog label match |
| clean_images/khillar | Khillari | 127 | 0 | excluded: not in explicit include list |
| clean_images/konkan_kapila |  | 11 | 0 | excluded: no catalog label match |
| clean_images/kosali |  | 10 | 0 | excluded: no catalog label match |
| clean_images/krishna_valley | Krishna Valley | 150 | 0 | excluded: not in explicit include list |
| clean_images/ladakhi |  | 10 | 0 | excluded: no catalog label match |
| clean_images/lakhimi |  | 10 | 0 | excluded: no catalog label match |
| clean_images/limousin |  | 5 | 0 | excluded: no catalog label match |
| clean_images/luit_(swamp) |  | 10 | 0 | excluded: no catalog label match |
| clean_images/malnad_gidda |  | 117 | 0 | excluded: no catalog label match |
| clean_images/malvi | Malvi | 10 | 0 | excluded: not in explicit include list |
| clean_images/marathwadi |  | 10 | 0 | excluded: no catalog label match |
| clean_images/mehsana | Mehsana | 105 | 0 | excluded: not in explicit include list |
| clean_images/mewati |  | 18 | 0 | excluded: no catalog label match |
| clean_images/motu |  | 10 | 0 | excluded: no catalog label match |
| clean_images/nagori |  | 99 | 0 | excluded: no catalog label match |
| clean_images/nagpuri | Nagpuri | 197 | 0 | excluded: not in explicit include list |
| clean_images/nari |  | 14 | 0 | excluded: no catalog label match |
| clean_images/nili_ravi | Nili-Ravi | 99 | 0 | excluded: not in explicit include list |
| clean_images/nimari | Nimari | 94 | 0 | excluded: not in explicit include list |
| clean_images/ongole | Ongole | 201 | 0 | excluded: not in explicit include list |
| clean_images/pandharpuri | Pandharpuri | 10 | 0 | excluded: not in explicit include list |
| clean_images/poda_thurpu |  | 10 | 0 | excluded: no catalog label match |
| clean_images/ponwar |  | 10 | 0 | excluded: no catalog label match |
| clean_images/pulikulam |  | 135 | 0 | excluded: no catalog label match |
| clean_images/punganur |  | 10 | 0 | excluded: no catalog label match |
| clean_images/purnea |  | 10 | 0 | excluded: no catalog label match |
| clean_images/rathi | Rathi | 159 | 0 | excluded: not in explicit include list |
| clean_images/red_dane |  | 167 | 0 | excluded: no catalog label match |
| clean_images/red_kandhari |  | 10 | 0 | excluded: no catalog label match |
| clean_images/red_sindhi | Red Sindhi | 172 | 0 | excluded: not in explicit include list |
| clean_images/shweta_kapila |  | 8 | 0 | excluded: no catalog label match |
| clean_images/siri |  | 12 | 0 | excluded: no catalog label match |
| clean_images/surti | Surti | 74 | 0 | excluded: not in explicit include list |
| clean_images/thutho |  | 10 | 0 | excluded: no catalog label match |
| clean_images/toda |  | 134 | 0 | excluded: no catalog label match |
| clean_images/umblachery |  | 86 | 0 | excluded: no catalog label match |
| clean_images/vechur |  | 150 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Alambadi |  | 99 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Amritmahal | Amritmahal | 94 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Ayrshire |  | 234 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Banni |  | 109 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Bargur |  | 94 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Bhadawari | Bhadawari | 86 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Brown_Swiss |  | 225 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Dangi |  | 82 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Deoni | Deoni | 99 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Guernsey |  | 119 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Hariana | Hariana | 130 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Holstein_Friesian |  | 328 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Jaffrabadi | Jaffarabadi | 102 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Jersey |  | 203 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Kangayam | Kangayam | 91 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Kankrej |  | 179 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Kasargod |  | 95 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Kenkatha |  | 55 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Kherigarh |  | 36 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Khillari | Khillari | 113 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Krishna_Valley | Krishna Valley | 136 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Malnad_gidda |  | 107 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Mehsana | Mehsana | 95 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Nagori |  | 89 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Nagpuri | Nagpuri | 187 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Nili_Ravi | Nili-Ravi | 89 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Nimari | Nimari | 84 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Ongole | Ongole | 191 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Pulikulam |  | 125 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Rathi | Rathi | 149 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Red_Dane |  | 167 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Red_Sindhi | Red Sindhi | 166 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Surti | Surti | 64 | 0 | excluded: not in explicit include list |
| Indian_bovine_breeds/Toda |  | 124 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Umblachery |  | 76 | 0 | excluded: no catalog label match |
| Indian_bovine_breeds/Vechur |  | 140 | 0 | excluded: no catalog label match |
