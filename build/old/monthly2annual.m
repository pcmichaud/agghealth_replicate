function [annual] = monthly2annual(monthly)
%monthly = 
%de janvier à décembre
%matrice avec Tm lignes et N colonnes

Tm = size(monthly,1);
N = size(monthly,2);
Ty = round(Tm/12);
annual=zeros(Ty,N);

for kk=1:N
    for jj=1:Ty
        annual(jj,kk)=mean(monthly(1+(jj-1)*12:jj*12,kk));
    end    
end


end

