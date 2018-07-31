---
layout: post
title: Custom Website in 30 Minutes
description: Setup and Build a Web Site on a Custom Domain
comments: true
tweets: true
---

First things first: bite the bullet and buy a custom
domain. The simplest route I've found in years is <a
href="http://domains.google.com" target="_blank">Google
Domains</a>. Prices are competitive and the interface is
intuitive. I recommend choosing to keep the registrar info
private during registration, it cuts down on spam and won't
expose personal info to an unknown audience.

<a data-fancybox="gallery" href="/img/galaxy.jpg">
   <img class="img-fluid border" src="http://www.kcavagnolo.com/img/gdomains.png" alt="DNS Setup">
</a>

One more step I need to describe is building a local instance
of github pages and projects w/ jekyll, and installing
jekyll-bootstrap to handle plumbing/subdir instance of a blog
(like this one). The shit that's still broken:

* ipython integration
* plotly integration
* native code highlighting, like the below

<p><a href="#" class="tooltip" title="Tooltip">This link</a></p>

{% highlight python linenos %}
def build_model(X, X_test, y):
    model = DecisionTreeClassifier()
    samples = sc.parallelize(Bootstrap(y.size))
    vote_tally = samples.map(lambda (index, _):
    model.fit(X[index], y[index]).predict(X_test)
    y_estimate_vote = np.argmax(vote_tally, axis = 1)
    return accuracy_score(y_estimate_vote)
{% endhighlight %}
