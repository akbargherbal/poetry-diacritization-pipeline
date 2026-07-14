```python
#
```


```python
import pandas as pd
import regex as re
import os
from pathlib import Path
```


```python
from pyarud.processor import ArudhProcessor
poem_processor = ArudhProcessor()
```


```python
meter = 'ramal'
verses = [
{"verse_id": "1919-4", "meter": "ramal", "sadr": "مَلْعَبُ الْأَيَّامِ إِلَّا أَنَّهُ", "ajuz": "لَيْسَ حَظُّ الْجَدِّ مِنْهُ بِالْقَلِيلِ"},
{"verse_id": "1919-6", "meter": "ramal", "sadr": "وَائْتَنَفْنَا فِي ذُرَاهَا دَوْلَةً", "ajuz": "رُكْنُهَا السُّؤْدَدُ وَالْمَجْدُ الْأَثِيلُ"},
{"verse_id": "1919-5", "meter": "ramal", "sadr": "شَهِدَ النَّاسُ بِهَا عَائِدَةً", "ajuz": "وَشَجَى الْأَجْيَالِ مِنْ فَرْدَيِ الْهُدَيْلِ" }

]
verses = [(verse['sadr'], verse['ajuz']) for verse in verses]

poem_processor.process_poem(verses=verses, meter_name=meter)

```




    {'meter': 'ramal',
     'verses': [{'verse_index': 0,
       'sadr_text': 'مَلْعَبُ الْأَيَّامِ إِلَّا أَنَّهُ',
       'ajuz_text': 'لَيْسَ حَظُّ الْجَدِّ مِنْهُ بِالْقَلِيلِ',
       'input_pattern': '101101010110101011010110101011011011010',
       'best_ref_pattern': '101101010110101011010110101011011011010',
       'score': 1.0,
       'sadr_analysis': [{'foot_index': 0,
         'expected_pattern': '1011010',
         'actual_segment': '1011010',
         'score': 1.0,
         'status': 'ok'},
        {'foot_index': 1,
         'expected_pattern': '1011010',
         'actual_segment': '1011010',
         'score': 1.0,
         'status': 'ok'},
        {'foot_index': 2,
         'expected_pattern': '10110',
         'actual_segment': '10110',
         'score': 1.0,
         'status': 'ok'}],
       'ajuz_analysis': [{'foot_index': 0,
         'expected_pattern': '1011010',
         'actual_segment': '1011010',
         'score': 1.0,
         'status': 'ok'},
        {'foot_index': 1,
         'expected_pattern': '101101',
         'actual_segment': '101101',
         'score': 1.0,
         'status': 'ok'},
        {'foot_index': 2,
         'expected_pattern': '1011010',
         'actual_segment': '1011010',
         'score': 1.0,
         'status': 'ok'}]},
      {'verse_index': 1,
       'sadr_text': 'وَائْتَنَفْنَا فِي ذُرَاهَا دَوْلَةً',
       'ajuz_text': 'رُكْنُهَا السُّؤْدَدُ وَالْمَجْدُ الْأَثِيلُ',
       'input_pattern': '101101010110101011010110101110101011010',
       'best_ref_pattern': '101101010110101011010110101110101011010',
       'score': 1.0,
       'sadr_analysis': [{'foot_index': 0,
         'expected_pattern': '1011010',
         'actual_segment': '1011010',
         'score': 1.0,
         'status': 'ok'},
        {'foot_index': 1,
         'expected_pattern': '1011010',
         'actual_segment': '1011010',
         'score': 1.0,
         'status': 'ok'},
        {'foot_index': 2,
         'expected_pattern': '10110',
         'actual_segment': '10110',
         'score': 1.0,
         'status': 'ok'}],
       'ajuz_analysis': [{'foot_index': 0,
         'expected_pattern': '1011010',
         'actual_segment': '1011010',
         'score': 1.0,
         'status': 'ok'},
        {'foot_index': 1,
         'expected_pattern': '111010',
         'actual_segment': '111010',
         'score': 1.0,
         'status': 'ok'},
        {'foot_index': 2,
         'expected_pattern': '1011010',
         'actual_segment': '1011010',
         'score': 1.0,
         'status': 'ok'}]},
      {'verse_index': 2,
       'sadr_text': 'شَهِدَ النَّاسُ بِهَا عَائِدَةً',
       'ajuz_text': 'وَشَجَى الْأَجْيَالِ مِنْ فَرْدَيِ الْهُدَيْلِ',
       'input_pattern': '1110101110101110111010101101011011010',
       'best_ref_pattern': '111010111010111011101010110101011010',
       'score': 0.93,
       'sadr_analysis': [{'foot_index': 0,
         'expected_pattern': '111010',
         'actual_segment': '111010',
         'score': 1.0,
         'status': 'ok'},
        {'foot_index': 1,
         'expected_pattern': '111010',
         'actual_segment': '111010',
         'score': 1.0,
         'status': 'ok'},
        {'foot_index': 2,
         'expected_pattern': '1110',
         'actual_segment': '1110',
         'score': 1.0,
         'status': 'ok'}],
       'ajuz_analysis': [{'foot_index': 0,
         'expected_pattern': '111010',
         'actual_segment': '111010',
         'score': 1.0,
         'status': 'ok'},
        {'foot_index': 1,
         'expected_pattern': '1011010',
         'actual_segment': '1011010',
         'score': 1.0,
         'status': 'ok'},
        {'foot_index': 2,
         'expected_pattern': '1011010',
         'actual_segment': '1101101',
         'score': 0.4,
         'status': 'broken'},
        {'foot_index': 3,
         'expected_pattern': '',
         'actual_segment': '0',
         'score': 0,
         'status': 'extra_bits'}]}]}




```python
df = pd.read_pickle('./SAMPLE_POEMS.pkl')
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>POET_NAME</th>
      <th>poem_no</th>
      <th>batch_no</th>
      <th>POET_RANK</th>
      <th>meter</th>
      <th>DATA</th>
      <th>BATCH_SIZE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>مهيار الديلمي</td>
      <td>65601</td>
      <td>4</td>
      <td>70</td>
      <td>wafer</td>
      <td>[{'verse_id': '65601_048', 'sadr': 'وخفت عليك ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>احمد زكي ابو شادي</td>
      <td>132023</td>
      <td>1</td>
      <td>79</td>
      <td>khafeef</td>
      <td>[{'verse_id': '132023_012', 'sadr': 'للذى ظل ف...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>احمد شوقي</td>
      <td>1980</td>
      <td>2</td>
      <td>10</td>
      <td>ramal</td>
      <td>[{'verse_id': '1980_024', 'sadr': 'ما الذي رد ...</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ابن سهل الاندلسي</td>
      <td>58959</td>
      <td>2</td>
      <td>76</td>
      <td>mutakareb</td>
      <td>[{'verse_id': '58959_024', 'sadr': 'همام محاري...</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>مهيار الديلمي</td>
      <td>65481</td>
      <td>6</td>
      <td>70</td>
      <td>rajaz</td>
      <td>[{'verse_id': '65481_072', 'sadr': 'لها بطون ا...</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>احمد محرم</td>
      <td>132727</td>
      <td>2</td>
      <td>86</td>
      <td>kamel</td>
      <td>[{'verse_id': '132727_024', 'sadr': 'وإذا النف...</td>
      <td>12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>لسان الدين بن الخطيب</td>
      <td>118432</td>
      <td>3</td>
      <td>44</td>
      <td>mutadarak</td>
      <td>[{'verse_id': '118432_036', 'sadr': 'فالرمح ين...</td>
      <td>12</td>
    </tr>
    <tr>
      <th>7</th>
      <td>محيي الدين بن عربي</td>
      <td>9299</td>
      <td>3</td>
      <td>40</td>
      <td>baseet</td>
      <td>[{'verse_id': '9299_036', 'sadr': 'عين صحيح جل...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ابن هانء الاندلسي</td>
      <td>59230</td>
      <td>0</td>
      <td>39</td>
      <td>taweel</td>
      <td>[{'verse_id': '59230_000', 'sadr': 'سرى وجناح ...</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
