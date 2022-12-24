async function getResult(event) {
    event.preventDefault();
    let input=document.getElementById('input');
    if (input.value=="") {
        window.alert("Please Enter Your Sentiment First");
    } else {
        window.alert("please Wait We are going to Perform some Machine Learning Task.......");
        try {
            let response=await fetch("/getresult",{
                method:"POST",
                headers:{
                    "Content-Type":"Application/json"
                },
                body:JSON.stringify({
                    review:input.value
                })
            });
            let result=await response.json();
            console.log(result);
            window.alert("result is fetched and shown below")
            let positiveResult=document.getElementById('positiveResult')
            let nagativeResult=document.getElementById('nagativeResult')
            let defaultResult=document.getElementById('default')
            let neutralResult=document.getElementById('neutralResult')
            if (!!result.data) {
                if (result.data=="Good") {
                    positiveResult.style.display='flex';
                    nagativeResult.style.display='none';
                    defaultResult.style.display='none';
                    neutralResult.style.display='none';
                } else if (result.data=="Bad") {
                    positiveResult.style.display='none';
                    nagativeResult.style.display='flex';
                    defaultResult.style.display='none';
                    neutralResult.style.display='none';
                }else{
                    positiveResult.style.display='none';
                    nagativeResult.style.display='none';
                    neutralResult.style.display='flex';
                    defaultResult.style.display='none';
                }
            } else {
                positiveResult.style.display='none';
                nagativeResult.style.display='none';
                defaultResult.style.display='flex';
                neutralResult.style.display='none';
                
            }
        } catch (error) {
            console.log(error);
        }
    }
}