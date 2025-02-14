function openTab(evt, tabName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}

const defaultOpenElement = document.getElementById("defaultOpen");
if (defaultOpenElement) {
  defaultOpenElement.click();
}

$(document).ready(function(){
  $('.testimonial').slick({
  slidesToShow: 3,
  dots:true,
  centerMode: true,
  });
});


var navOpen = false;

function toggleNav() {
  if (navOpen) {
    closeNav();
  } else {
    openNav();
  }
}

function openNav() {
  document.getElementById("mySidenav").style.right = "0";
  navOpen = true;
}

function closeNav() {
  document.getElementById("mySidenav").style.right = "-281px";
  navOpen = false;
}


function installCode() {
  var elemento = document.getElementById("install");
  var selecao = window.getSelection();
  var intervalo = document.createRange();
  intervalo.selectNodeContents(elemento);
  selecao.removeAllRanges();
  selecao.addRange(intervalo);
  document.execCommand("copy");
  selecao.removeAllRanges();
}


document.addEventListener("DOMContentLoaded", function () {
  const logo = document.querySelector('a.md-header__button.md-logo');

  if (logo) {
    const newLogo = document.createElement("a");
    newLogo.href = "./";
    newLogo.className = "md-header__button md-logo";
    newLogo.setAttribute("aria-label", "SysIdentPy");
    newLogo.setAttribute("data-md-component", "logo");
    newLogo.setAttribute("title", "SysIdentPy");

    const img = document.createElement("img");
    img.src = "overrides/assets-2.0/img/logotype-sysidentpy.svg";
    img.alt = "Logotype SysIdentPy";

    newLogo.appendChild(img);
    logo.replaceWith(newLogo);
  }
});

