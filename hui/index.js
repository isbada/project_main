delThisLi = function(id) {
  var obj = document.getElementById(id.id);
  if (obj != null)
    obj.parentNode.removeChild(obj);
}
$(document).ready(function () {
  var array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
  function C(arr, num) {
    var r = [];
    (function f(t, a, n) {
      if (n == 0) {
        return r.push(t);
      }
      for (var i = 0, l = a.length; i <= l - n; i++) {
        f(t.concat(a[i]), a.slice(i + 1), n - 1);
      }
    })([], arr, num);
    return r;
  }
  var numArr6 = C(array, 6)
  var numArr5 = C(array, 5);
  var numArr = numArr5; // //存储了所有可能的组合
  // 5  or 6
  isfive = true
  $('input[name="fx_radio"]').change(function(){
    fxValue = parseInt($('input[name="fx_radio"]:checked').val()[0])
    if(fxValue == 6){
      isfive = false
      numArr = numArr6
      $('.six').css('display', 'inline-block')
      $('.five').hide()
      
    } else {
      isfive = true
      numArr = numArr5
      $('.five').css('display', 'inline-block')
      $('.six').hide()
    }
  })
  $('#filterbtn').click(function () {
    hasDuolu = $('#duolu')[0].checked
    hasOddEven = $('#oddEven')[0].checked
    hasBigSmall = $('#bigSmall')[0].checked
    hasZhihe = $('#zhihe')[0].checked
    hasDuiChong = $('#duichong')[0].checked
    hasNumber = $('#hasNumber')[0].checked
    hasNotNumber = $('#hasNotNumber')[0].checked
    filterResult = numArr

    // 多路筛选
    if (hasDuolu) {
      road1 = [3, 6, 9]
      road2 = [1, 4, 7, 10]
      road3 = [2, 5, 8, 11]
      filterNumArrry = []
      if(isfive){
        num1 = parseInt($('.five input[name="dl_radio"]:checked').val()[0])
        num2 = parseInt($('.five input[name="dl_radio"]:checked').val()[1])
        num3 = parseInt($('.five input[name="dl_radio"]:checked').val()[2])
      } else {
        num1 = parseInt($('.six input[name="dl_radio"]:checked').val()[0])
        num2 = parseInt($('.six input[name="dl_radio"]:checked').val()[1])
        num3 = parseInt($('.six input[name="dl_radio"]:checked').val()[2])
      }
      
      for (item of filterResult) {
        countRoad1 = 0
        countRoad2 = 0
        countRoad3 = 0
        for (road1Item of road1) {
          if (item.indexOf(road1Item) != -1) {
            countRoad1++
          }
        }
        for (road2Item of road2) {
          if (item.indexOf(road2Item) != -1) {
            countRoad2++
          }
        }
        for (road3Item of road3) {
          if (item.indexOf(road3Item) != -1) {
            countRoad3++
          }
        }
        if (countRoad1 == num1 && countRoad2 == num2 && countRoad3 == num3) {
          filterNumArrry.push(item)
        }
      }
      filterResult = filterNumArrry
    }
    // 奇偶
    // debugger
    if (hasOddEven) {
      debugger
      filterNumArrry = []
      if(isfive){
        oddNum = parseInt($('.five input[name="jo_radio"]:checked').val()[0])
        evenNum = parseInt($('.five input[name="jo_radio"]:checked').val()[1])
      } else {
        oddNum = parseInt($('.six input[name="jo_radio"]:checked').val()[0])
        evenNum = parseInt($('.six input[name="jo_radio"]:checked').val()[1])
      }
      
      oddArray = [1,3,5,7,9,11]
      evenArray = [2,4,6,8,10]
      for (item of filterResult) {
        countodd = 0
      counteven = 0
        for(t of item){
          if(oddArray.indexOf(t)!=-1)
            countodd++  
          if(evenArray.indexOf(t)!=-1)
            counteven++
        }
        if (counteven == evenNum && countodd == oddNum) {
          filterNumArrry.push(item)
        }
      }
      
      filterResult = filterNumArrry
    }
    //  大小数
    if (hasBigSmall) {
      filterNumArrry = []
      if(isfive){
        bigNum = parseInt($('.five input[name="dx_radio"]:checked').val()[0])
        smallNum = parseInt($('.five input[name="dx_radio"]:checked').val()[1])
      } else {
        bigNum = parseInt($('.six input[name="dx_radio"]:checked').val()[0])
        smallNum = parseInt($('.six input[name="dx_radio"]:checked').val()[1])
      }
      
      bigArray = [6, 7, 8, 9, 10, 11]
      smallArray = [1, 2, 3, 4, 5]
      for (item of filterResult) {
        countbig = 0
        countsmall = 0
        for(t of item){
          if(bigArray.indexOf(t)!=-1)
            countbig++  
          if(smallArray.indexOf(t)!=-1)
            countsmall++
        }
        if (countsmall == smallNum && countbig == bigNum) {
          filterNumArrry.push(item)
        }
      }
      filterResult = filterNumArrry
    }
    //  质合数
    if (hasZhihe) {
      filterNumArrry = []
      if(isfive){
        zhiNum = parseInt($('.five input[name="zh_radio"]:checked').val()[0])
        heNum = parseInt($('.five input[name="zh_radio"]:checked').val()[1])
      } else {
        zhiNum = parseInt($('.six input[name="zh_radio"]:checked').val()[0])
        heNum = parseInt($('.six input[name="zh_radio"]:checked').val()[1])
      }

      zhiArray = [1, 2, 3, 5, 7, 11]
      heArray = [4, 6, 8, 9, 10]
      for (item of filterResult) {
        countzhi = 0
          counthe = 0
          for(t of item){
            if(zhiArray.indexOf(t)!=-1)
              countzhi++  
            if(heArray.indexOf(t)!=-1)
              counthe++
          }
        if (counthe == heNum && countzhi == zhiNum) {
          filterNumArrry.push(item)
        }
      }
      filterResult = filterNumArrry
    }
    // 对冲
    if (hasDuiChong) {
      filterNumArrry = []
      number1 = parseInt($('#dc1')[0].value)
      number2 = parseInt($('#dc2')[0].value)
      for (item of filterResult) {
        if (!(item.indexOf(number1) != -1 && item.indexOf(number2) != -1)) {
          filterNumArrry.push(item)
        }
      }
      filterResult = filterNumArrry
    }
    // 必有数字筛选
    if (hasNumber) {
      filterNumArrry = []
      number = parseInt($('#bysz')[0].value)
      for (item of filterResult) {
        if (item.indexOf(number) != -1) {
          filterNumArrry.push(item)
        }
      }
      filterResult = filterNumArrry
    }
    // 必无数字筛选
    if (hasNotNumber) {
      filterNumArrry = []
      number = parseInt($('#bwsz')[0].value)
      for (item of filterResult) {
        if (item.indexOf(number) == -1) {
          filterNumArrry.push(item)
        }
      }
      filterResult = filterNumArrry
    }
    // debugger
    
    document.getElementById('allNum').innerHTML = ''
    for (var i = 0; i < filterResult.length; i++) {
      var id = 'li' + (parseInt(i) + 1);
      // document.getElementById('allNum').innerHTML+= (parseInt(i)+1) + '\t\t' +numArr[i].join('-')+"<br>"
      document.getElementById('allNum').innerHTML +=
        '<li id=' + id + '>' + filterResult[i].join('-') +
        '&nbsp;&nbsp;&nbsp;&nbsp;' +
        '<button onclick=delThisLi(' + id + ')>删除本条</button>' +
        "</li>"
    };
    
  })

})