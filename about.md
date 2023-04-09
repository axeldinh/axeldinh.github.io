---
title: About
layout: page
---
![Profile Image]({% if site.external-image %}{{ site.picture }}{% else %}{{ site.url }}/{{ site.picture }}{% endif %})

My name is Axel, I am a fresh graduate of <a href="https://www.epfl.ch/"><u>EPFL</u></a>'s Computational Science and Engineering Master program and I am highly interested in Deep Learning
and its many applications. Having a bachelor in Mathematics, I am not scared of theoretical problems,
such as differential equations or numerical methods.

<p> I just finished working on a project about Deep Learning on MRI segmentation at EPFL's <a href="https://www.epfl.ch/labs/cvlab/"><u>Computer Vision Lab</u></a>. 
As this project ended on March 2023, I am looking for a new challenge. If my profile picked your interest, 
do not hesitate to contact me! </p>

<h2>Skills</h2>

<ul class="skill-list">
	<li>Pytorch</li>
	<li>Pandas</li>
	<li>Scikit-Learn</li>
	<li>Slurm</li>
	<li>Git</li>
	<li>C++</li>
	<li>Matlab</li>
	<li>LaTex</li>
	<li>Test-Driven Developpement</li>
</ul>

<h2>Projects</h2>

<ul>
{% for post in site.posts %}
	{% if post.projects %}
		<li> <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a> </li>
	{% endif %}
{% endfor %}
</ul>
