<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/DaehwanKimLab/Chromosome-ReAD">
    <img src="images/ReAD.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Chromosome-ReAD</h3>

  <p align="center">
    A deep convolutional neural network for Recurrent Abnormality Detection in chromosomes!
    <br />
    <a href="https://github.com/DaehwanKimLab/Chromosome-ReAD"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/DaehwanKimLab/Chromosome-ReAD">View Demo</a>
    ·
    <a href="https://github.com/DaehwanKimLab/Chromosome-ReAD">Report Bug</a>
    ·
    <a href="https://github.com/DaehwanKimLab/Chromosome-ReAD">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#manuscript">Manuscript</a>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- MANUSCRIPT --> 
## Manuscript
This work has been published in Bioinformatics, an Oxford Academic journal. Please check out [the paper](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btab822/6454943) and reach out with any questions or comments! 

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

<!-- There are many great README templates available on GitHub, however, I didn't find one that really suit my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need -- I think this is it.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should element DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have have contributed to expanding this template!

A list of commonly used resources that I find helpful are listed in the acknowledgements. -->

There are many great neural networks for classifying chromosomes, however, a great deal of clinical cytogenetics involves analyzing samples with abnormal chromosomes. So, we created Chromosome Recurrent Abnormality Detector (ReAD) to automatically classify many important abnormalities found in patients with blood cancer. 

Here's why: 
* Currently, cytogeneticists spend the majority of their time manually classifying chromosomes but can only inspect a tiny fraction of the cells collected from imaging. 
* Recurrent abnormalities can heavily inform the diagnosis, prognisis, and treatment plan. 
* An automated classifier could screen hundreds of cells for abnormalities and flag cells requiring further inspection. 

Certainly, this neural network does not cover all types of recurrent abnormalities in hematopathology. So I'll be updating the model in the near future to incorporate more. If you have your own dataset, you may also suggest changes by forking this repo and creating a pull request or opening an issue. 

### Built With

<!-- This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Bootstrap](https://getbootstrap.com)
* [JQuery](https://jquery.com)
* [Laravel](https://laravel.com) -->
* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [FastAI](https://www.fast.ai/)
 


<!-- GETTING STARTED -->
## Getting Started

<!-- This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps. -->

### Prerequisites

<ul>
  <li> python 3.7 </li>
  <li> fastai 2.1.4 </li>
  <li> fastai2 0.0.30 </li>
  <li> fastasi2-extensions 0.0.31 </li>
  <li> torch 1.7.0 </li>
  <li> jupyter 1.0.0 </li>
  <li> numpy 1.19.4 </li>
</ul>

A dataset of individual chromosome images. 

### Installation

Clone the repository: 
`git clone https:github.com/DaehwanKimLab/Chromosome-ReAD`. 

<!-- Prepare dataset: `cd Chromosome-ReAD/resources/feb2021_abclass`. \
Perform for each [train, val, test] directories:\
`unzip []` --> 

<!-- USAGE EXAMPLES -->
## Usage

<!-- Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_ -->

The <experiments> directory contains 4 Jupyter notebooks, one for each convolutional neural network architecture tested, that can be ran to repeat training network from scratch and inferring on the test set. If you would like to preserve the orginal experimental results but would like to rerun the experiments yourself, simply make a copy a notebook and run the cells in the duplicate. 

<!-- ROADMAP -->
## Roadmap

See the [open issues](https:github.com/DaehwanKimLab/Chromosome-ReAD/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@and_m_cox](https://twitter.com/and_m_cox) - Andrew.Cox@UTSouthwestern.edu

Project Link: [https://github.com/DaehwanKimLab/Chromosome-ReAD](https://github.com/DaehwanKimLab/Chromosome-ReAD)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
UT Southwestern Bioinformatics and Pathology Departments. 
<!-- * [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com) -->





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/DaehwanKimLab/Chromosome-ReAD.svg?style=for-the-badge
[contributors-url]: https://github.com/DaehwanKimLab/Chromosome-ReAD/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/DaehwanKimLab/Chromosome-ReAD.svg?style=for-the-badge
[forks-url]: https://github.com/DaehwanKimLab/Chromosome-ReAD/network/members
[stars-shield]: https://img.shields.io/github/stars/DaehwanKimLab/Chromosome-ReAD.svg?style=for-the-badge
[stars-url]: https://github.com/DaehwanKimLab/Chromosome-ReAD/stargazers
[issues-shield]: https://img.shields.io/github/issues/DaehwanKimLab/Chromosome-ReAD.svg?style=for-the-badge
[issues-url]: https://github.com/DaehwanKimLab/Chromosome-ReAD/issues
[license-shield]: https://img.shields.io/github/license/DaehwanKimLab/Chromosome-ReAD.svg?style=for-the-badge
[license-url]: https://github.com/DaehwanKimLab/Chromosome-ReAD/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/andrew-m-cox/
[product-screenshot]: images/screenshot.png
