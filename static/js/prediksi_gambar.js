$(function() {
  
  // --------------------------------------
  
  $("#prediksi_submit").click(function(e) {
    e.preventDefault();
    var file_data = $('#input_gambar').prop('files')[0];   
    var pics_data = new FormData();                  
    pics_data.append('file', file_data);
	

    setTimeout(function() {
	  try {
			$.ajax({
			  url:"/api/deteksi",
			  type:"POST",
			  data:pics_data,
			  processData: false,
			  contentType: false,
			  success:function(res){
				res_data_prediksi = res['prediksi']
				res_gambar_prediksi = res['gambar_prediksi']
			    generate_prediksi(res_data_prediksi, res_gambar_prediksi); 
			  }
			});
		}
		catch(e) {
			console.log(e);
		} 
    }, 1000)
    
  })
  
  // --------------------------------------
  
  
  function generate_prediksi(data_prediksi, image_prediksi) {
	var str="";
	
	if(image_prediksi == "(none)") {
		str += "<h4>Silahkan masukkan file gambar (.jpg)</h4>";
	}
	else {
		str += "<img src='" + image_prediksi + "' width=\"200\"></img>"
		str += "<h4>" + data_prediksi + "</h4>";
	}
	$("#hasil_prediksi").html(str);
  }  
})
  
