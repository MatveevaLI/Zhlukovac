Máme 2D priestor, ktorý má rozmery X a Y, v intervaloch od -5000 do +5000. Tento 2D priestor vplneny 20 bodmi, pričom každý bod má náhodne zvolenu polohu pomocou súradníc X a Y. Každý bod má unikátne súradnice.
Po vygenerovaní 20 náhodných bodov vygenerované ďalších 20000 bodov, avšak tieto body nebudú generované úplne náhodne, ale nasledovným spôsobom:
1.	Náhodne vybraný jeden zo všetkých doteraz vytvorených bodov v 2D priestore.
2.	Vygenerované náhodne číslo X_offset v intervale od -100 do +100
3.	Vygenerované náhodne číslo Y_offset v intervale od -100 do +100
4.	Pridaný nový bod do 2D priestoru, ktorý bude mať súradnice ako náhodne vybraný bod v kroku 1, pričom tieto súradnice budú posunuté o X_offset a Y_offset
Naprogramovaný zhlukovač pre 2D priestor, ktorý analyzuje 2D priestor so všetkými jeho bodmi a delí tento priestor na k zhlukov (klastrov). Implementované rôzne verzie zhlukovača, konkrétne týmito algoritmami:
•	k-means, kde stred je centroid
•	k-means, kde stred je medoid
•	aglomeratívne zhlukovanie, kde stred je centroid
•	divízne zhlukovanie, kde stred je centroid
