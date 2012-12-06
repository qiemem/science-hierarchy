from django.conf.urls import patterns, include, url

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    url(r'^home/', 'ml_site.packages.views.home'),
    url(r'^done/', 'ml_site.packages.views.done'),
    url(r'^question/(?P<q_id>\d+)/$', 'ml_site.packages.views.question'),
    url(r'^question/(?P<q_id>\d+)/vote/', 'ml_site.packages.views.vote'),
    # url(r'^ml_site/question/(?P<q_id>\d+)', include('ml_site.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    url(r'^admin/', include(admin.site.urls)),
)
