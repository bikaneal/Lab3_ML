# Лабораторная работа #3. РАЗРАБОТКА СИСТЕМЫ МАШИННОГО ОБУЧЕНИЯ.
Отчет по лабораторной работе #3 выполнил(а):
- Ли Александр Альбертович
- Х21IT_AI-01BL

Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨


## Цель работы
Познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity. В данной лабораторной работе мы создадим ML-агент и будем тренировать
нейросеть, задача которой будет заключаться в управлении шаром. Задача шара заключается в том, чтобы оставаясь на плоскости находить кубик, смещающийся в заданном случайном диапазоне координат.

## Задание 1
### Реализовать систему машинного обучения в связке Python – Google-Sheets – Unity.

Установка необходимых средств и настройка системы машинного обучения выполнились успешно.

![sphere (1)](https://user-images.githubusercontent.com/78469125/195573894-7acb9afa-5400-48a2-9e03-5685da1e5e6b.gif)

Установка дополнительных одинаковых полей позволяют ускорить обучение ввиду увеличению итераций.

![3 things](https://user-images.githubusercontent.com/78469125/195575458-60b692ac-a1a4-4709-bb2f-a5ae36713b9a.PNG)

Шаг Step стал увеличиваться быстрее с 54 полями:

![40k rotations](https://user-images.githubusercontent.com/78469125/195575645-f3f278f9-2d1e-437d-b29e-d23bc85aed8c.PNG)

После, возвращаясь к единой модели, заметно улучшилась ловкость шара: он стал быстрее и точнее догонять куб. Движение стало плавным. Таким образом, машинное обучение позволило шару с меньшей задержкой достигать цели.



## Задание 2

### Подробно опишите каждую строку файла конфигурации нейронной сети, доступного в папке с файлами проекта по ссылке. Самостоятельно найдите информацию о компонентах Decision Requester, Behavior Parameters, добавленных на сфере.

```
behaviors:
  RollerBall:
  # proximal policy optimization. Данная функция использует нейронную сеть для приближения к идеальной функции, сопоставляемая с наилучшим действием агента
    trainer_type: ppo 
    # hyperparameters запуск обучения с задаваемыми параметрами
    hyperparameters:
     # batch_size количество опытов (описано ниже)
      batch_size: 10
     # buffer_size соответствует количеству опыта (наблюдения, действия и награды), необходимое для сбора до повторного обучения или обновления модели
      buffer_size: 100 
     # learning_rate соответствует силе алгоритма для поиска минимума функции потерь. Обычно это значение следует уменьшить, если тренировка нестабильна, а вознаграждение не увеличивается последовательно
      learning_rate: 3.0e-4 
     # beta соответствует силе параметра entropy regularization, отвечающая за рандомизацию РРО. Это позволяет агентам корректно исследовать пространство действий. Увеличение этого значения обеспечит выполнение большего количества случайных действий
      beta: 5.0e-4
     # epsilon соответствует допустимому порогу расхождения между старой и новой политиками при обновлении с градиентным спуском (методом нахождения локального минимума или максимума функции). Установка этого значения небольшим приведет к более стабильным обновлениям, но также замедлит процесс обучения
      epsilon: 0.2
     # lambd соответствует параметру лямбда, используемому при расчете Обобщенной оценки преимущества (GAE). Это можно рассматривать как то, насколько агент полагается на свою текущую оценку стоимости при расчете обновленной оценки стоимости. Низкие значения соответствуют тому, что вы больше полагаетесь на текущую оценку ценности (что может быть большой погрешностью), а высокие значения соответствуют тому, что вы больше полагаетесь на фактические вознаграждения, полученные в окружающей среде (что может быть высокой дисперсией). Параметр обеспечивает компромисс между ними, и правильное значение может привести к более стабильному процессу обучения
      lambd: 0.99
     # num_epoch — это количество проходов через буфер опыта во время градиентного спуска
      num_epoch: 3
     # тип алгоритма
      learning_rate_schedule: linear
    network_settings:
     # normalize соответствует тому, применяется ли нормализация к входным данным векторного наблюдения. Эта нормализация основана на текущем среднем значении и дисперсии векторного наблюдения. Нормализация может быть полезна в случаях со сложными задачами непрерывного управления, но может быть вредна при более простых задачах дискретного управления.
      normalize: false
     # hidden_units соответствует количеству единиц в каждом полностью подключенном слое нейронной сети. Для простых задач, где правильным действием является простая комбинация входных данных наблюдения, значение должно быть небольшим. Для задач, где действие представляет собой очень сложное взаимодействие между переменными наблюдения, значение должно быть больше.
      hidden_units: 128
     # num_layers соответствует количеству скрытых слоев, присутствующих после ввода наблюдения. Для простых задач меньшее количество слоев, скорее всего, будет обучаться быстрее и эффективнее. Для более сложных задач управления может потребоваться больше уровней.
      num_layers: 2
    reward_signals:
      extrinsic:
     # gamma соответствует коэффициенту будущих вознаграждений. Это можно рассматривать как то, насколько далеко в будущем агент должен заботиться о возможных вознаграждениях
        gamma: 0.99
        strength: 1.0
    max_steps: 500000
    # time_horizon соответствует количеству шагов сбора опыта для каждого агента до добавления его в буфер опыта. Когда этот предел достигается до окончания эпизода, используется оценка стоимости  для прогнозирования общего ожидаемого вознаграждения от текущего состояния агента. Таким образом, этот параметр является компромиссом между менее предвзятой, но более высокой оценкой дисперсии (long time horizon) и более предвзятой, но менее разнообразной оценкой (short time horizon)
    time_horizon: 64
    # summary_freq соответствует частоте обновления опыта агента
    summary_freq: 10000
```

Decision Requester вызывает алгоритм Agent.RequestDecision().
Behavior Parameters — параметры, определяющие, какую политику получит Агент.

## Задание 3
### Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и в первом задании, случайно изменять координаты на плоскости.

Выполнено. Ниже представлен гиф с демонстрацией работы модели. Также код.
![3уу](https://user-images.githubusercontent.com/78469125/196960631-62d67394-7491-4666-bb79-5dd9cfb71f24.gif)

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public GameObject Target;
    public GameObject Target2;
    private bool touch_target;
    private bool touch_target2;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.transform.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
        Target2.transform.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
        Target.SetActive(true);
        Target2.SetActive(true);
        touch_target = false;
        touch_target2 = false;
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.transform.localPosition);
        sensor.AddObservation(Target2.transform.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(touch_target);
        sensor.AddObservation(touch_target2);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.transform.localPosition);
        float distancetoTarget2 = Vector3.Distance(this.transform.localPosition, Target2.transform.localPosition);

        if(!touch_target & distanceToTarget < 1.42f)
        {
            touch_target = true;
            Target.SetActive(false);
        }

        if (!touch_target2 & distanceToTarget < 1.42f)
        {
            touch_target2 = true;
            Target2.SetActive(false);
        }

        if (touch_target & touch_target2)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            SetReward(-0.5f);
            EndEpisode();
        }
    }
}
```

## Выводы

Игровой баланс — формат взаимодействия между героем и персонажами, своего рода "равновесие" между этими взаимодействиями для максимально комфортного и реалистичного игрового опыта. Для этого необходимы точные математические расчеты, но пренебрежение самой игрой и развлечением ради математики не рекомендуется. Разнообразие NPC, их реалистичное поведение приводят к положительному игровому опыту. Собственно, машинное обучение является полезным инструментом для реализации "реалистичности" поведения персонажей, "обучая" их необходимым действиям. Например, в данной лабораторной работе мы обучали шарик таким образом, чтобы он максимально плавно передвигался от одного куба ко второму. После нескольких тысяч итераций шар начинает запоминать положения этих кубов и начинает их предсказывать, таким образом сокращая время передвижения и предоставляя максимально естественное поведение.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
