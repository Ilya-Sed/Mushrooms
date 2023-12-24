# Грибы. Съедобно или нет?
!["Manager prodgect"](https://img.shields.io/badge/%D0%9C%D0%B5%D0%BD%D0%B5%D0%B4%D0%B6%D0%B5%D1%80%20%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B0-%D0%90%D0%BD%D0%B4%D1%80%D0%B5%D0%B9%20%D0%9A%D1%85%D0%B0%D0%BB%D0%BE%D0%B2-blue
)
!["ML developer"](https://img.shields.io/badge/ML%20%D0%A0%D0%B0%D0%B7%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%87%D0%B8%D0%BA-%D0%94%D0%BC%D0%B8%D1%82%D1%80%D0%B8%D0%B9%20%D0%93%D0%BE%D0%BB%D0%BE%D0%B2%D0%B0%D1%87%D0%B5%D0%B2-orange
)
!["ML developer"](https://img.shields.io/badge/ML%20%D0%A0%D0%B0%D0%B7%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%87%D0%B8%D0%BA-%D0%98%D0%BB%D1%8C%D1%8F%20%D0%91%D0%B5%D1%86%D1%83%D0%BA%D0%B5%D0%BB%D0%B8-pink
)
!["ML developer"](https://img.shields.io/badge/ML_%D0%A0%D0%B0%D0%B7%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%87%D0%B8%D0%BA-%D0%98%D0%BB%D1%8C%D1%8F%20%D0%A1%D1%82%D0%BE%D1%80%D0%BE%D0%B6%D0%B5%D0%B2-yellow
)
!["fullstack developer"](https://img.shields.io/badge/%D0%A4%D1%83%D0%BB%D1%81%D1%82%D0%B5%D0%BA_%D0%A0%D0%B0%D0%B7%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%87%D0%B8%D0%BA-%D0%A1%D0%B5%D0%BC%D1%91%D0%BD_%D0%A8%D1%83%D0%BB%D1%8C%D0%B3%D0%B0-brightgreen
)
!["GitHub manager"](https://img.shields.io/badge/GitHub_%D0%9C%D0%B5%D0%BD%D0%B5%D0%B4%D0%B6%D0%B5%D1%80-%D0%98%D0%BB%D1%8C%D1%8F%20%D0%A1%D0%B5%D0%B4%D0%B5%D0%BB%D1%8C%D0%BD%D0%B8%D0%BA%D0%BE%D0%B2-red)


## О нашем проекте:

- [x] **Наша задача** - помочь росиянам в выборе грибов в лесу.
- [x] **Наш метод** - Создать чат бот, с помощью которого любой желающий сможет проверить, можно ли употреблять в пищу гриб, который нашли в лесу.
- [ ] [**Наш чат-бот**](pass) - Уже сейчас Вы можете проверить, возможно ли съесть гриб, что Вы нашли в лесу


## Немного статистики
В текущем году, по оценкам Рослесинфорга (структура Рослесхоза), объем собранных в российских лесах грибов побьет прошлогодний рекорд — в 2022 году в коммерческих целях в лесах собрали 834 т грибов.
<br>С начала 2023 года грибами отравились 108 росиян.
<br>[Газета известия](https://www.kommersant.ru/doc/6138423)

```python
# Статистика отравлений по московской области за 2023 год:
```
| Наименование гриба | Число отравившихся | Процент от общего числа |
|:-----------------------|:-------------:|------------:|
| Бледная поганка | 11 | 10,2 |
| Строчки| 21 | 19,4|
| Ложные опята| 9 | 8,4|
| Мухоморы| 3 | 2,7|
| Другие| 64 | 59,3|
![Статистика](C:\Users\79174\Desktop\Безымянный-1.pdf, 'Статистика отравлений')

## Для запуска ноутбука c моделями необходимо:
* Скачать [датасет](https://drive.google.com/drive/folders/1kn1HLN-Z_GG5leX4xJLa9waT1MQSvC9O?usp=sharing) для бинарной классификации
* Скачать [датасет](https://drive.google.com/drive/folders/1sUGHBw7gvrmB2ISi5D3lbAsQIvABTagc?usp=sharing) для классификации по типам
* Скачать [модель EfficientNetB7](https://drive.google.com/file/d/1rPa8nbNrb-bJ9r3EKP_ft8bQjESeF2n6/view?usp=sharing)
* Отредактировать ячейки ноутбука, содержащие пути к файлам в соответствии с вашей архитектурой проекта
* Всё готово
## Pickle файлы
* Бинарная [модель](https://drive.google.com/file/d/1ZJfF9WSIP8SogkLLUh9DPQ0rPJ0wOa4N/view?usp=sharing) в формате .pkl
* [Модель](https://drive.google.com/file/d/14_k28veTz0XibA50E8vNByuVxTWDoo9N/view?usp=sharing) для классификации по типам в формате .pkl
