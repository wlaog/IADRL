initprio1 = [];
initprio0 = [];
count_p1 = 0;
count_p0 = 0;
for i =1:299
    if Intdata(i).prio == 1
        count_p1 = count_p1+1;
        initprio1(count_p1,:) = [Intdata(i).motion(1,1:3),Intdata(i).motion(1,5:7)];
    else
        count_p0 = count_p0+1;
        initprio0(count_p0,:) = [Intdata(i).motion(1,1:3),Intdata(i).motion(1,5:7)];
    end
end
writematrix(initprio1,'initprio1.csv');
writematrix(initprio0,'initprio0.csv');
