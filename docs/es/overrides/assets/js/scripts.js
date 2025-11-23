function openTab(evt, tabName) {
  const tabcontent = document.getElementsByClassName("tabcontent");
  for (let i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  const tablinks = document.getElementsByClassName("tablinks");
  for (let i = 0; i < tablinks.length; i++) {
    tablinks[i].classList.remove("active");
  }
  const currentTab = document.getElementById(tabName);
  if (currentTab) {
    currentTab.style.display = "block";
  } else {
    console.warn(`Error: Not found tab ${tabName}`);
  }
  if (evt.currentTarget && evt.currentTarget.classList) {
    evt.currentTarget.classList.add("active");
  } else {
    console.warn("Error element.");
  }
}
window.openTab = openTab;

const defaultOpenElement = document.getElementById("defaultOpen");
if (defaultOpenElement) {
  defaultOpenElement.click();
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
window.installCode = installCode;

document.addEventListener("DOMContentLoaded", function () {
  var swiper = new Swiper(".mySwiper", {
    slidesPerView: 3,
    spaceBetween: 20,
    loop: true,
    autoplay: {
      delay: 2000,
      disableOnInteraction: false,
    },
    navigation: {
      nextEl: ".swiper-button-next",
      prevEl: ".swiper-button-prev",
    },
    pagination: {
      el: ".swiper-pagination",
      clickable: true,
    },
    breakpoints: {
      0: {
        slidesPerView: 1,
        navigation: false,
      },
      768: {
        slidesPerView: 1,
      },
      1200: {
        slidesPerView: 3,
      },
    },
  });
});
