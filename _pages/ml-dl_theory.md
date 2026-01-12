---
layout: single
permalink: /ml/dl_theory/
title: "DL Theory"
author_profile: false
show_excerpts: false
sidebar:
  nav: "ml"
---

{% assign ml_posts = site.categories.ml %}
{% for post in ml_posts %}
  {% if post.categories contains "dl_theory" %}
    {% include archive-single.html type="post" %}
  {% endif %}
{% endfor %}
