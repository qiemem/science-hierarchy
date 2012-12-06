from django.db import models

# Create your models here.
class Question(models.Model):
    # sample abstract
    prompt = models.CharField(max_length=5000)

    # control choice of abstract
    controlChoice = models.CharField(max_length=5000)

    # experiment choice abstract
    test = models.CharField(max_length=5000)

    # type of question
    # Either:
    #    -Distance metric validation: 0
    #    -Regular K-means validation: 1
    #    -Resampling K-means validation: 2
    choiceType = models.IntegerField()

    # Choice made by user
    #    -control: 1
    #    -test: 2
    selected = models.IntegerField()
