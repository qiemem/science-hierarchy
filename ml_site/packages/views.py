from django.http import HttpResponseRedirect, HttpResponse
from django.core.urlresolvers import reverse
from django.shortcuts import render_to_response, get_object_or_404
import random
from ml_site.packages.models import Question
from django.template import RequestContext

# home view
#    Explains the test and has a start test button
def home(request):
    return render_to_response('home.html')

# question site
def question(request,q_id):
    q = get_object_or_404(Question, pk=q_id)

    # Save prompt into X
    X = q.prompt

    # randomize choices
    r_int = random.randint(0,1)
    if r_int == 0:
        A = q.controlChoice
        B = q.test
        choiceOrder = 1
    else:
        B = q.controlChoice
        A = q.test
        choiceOrder = []

    # render site
    return render_to_response('question.html', {'q_id': q_id, 'X': X, 'A': A, 'B': B, 'choiceOrder': choiceOrder}, context_instance=RequestContext(request))

# form submission for vote
def vote(request, q_id):
    q = get_object_or_404(Question, pk=q_id)
    try:
        choice_id=request.POST['choice']
    except (KeyError):
        # Redisplay the poll voting form.
        return HttpResponseRedirect(reverse('ml_site.packages.views.question', args=(q_id,)))
    else:
        q.selected = int(choice_id)
        q.save()
        # redirect to the next question or to a done page
        if q.id+1 < 3:
            return HttpResponseRedirect(reverse('ml_site.packages.views.question', args=(q.id+1,)))
        else:
            return HttpResponseRedirect(reverse('ml_site.packages.views.done'))

# done page
def done(request):
    return render_to_response('done.html')
